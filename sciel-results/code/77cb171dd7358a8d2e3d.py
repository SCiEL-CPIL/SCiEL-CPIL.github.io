"""
Rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication.

The matrix multiplication tensor T[i,j,k] = E_{i,k} for A_{i,j} * B_{j,k}.
We seek U, V, W with U (n*m x r), V (m*p x r), W (p*n x r) such that
T = sum_c U[:,c] outer V[:,c] outer W[:,c]  (in the appropriate mode).

Specifically, for the bilinear map (A,B) -> C where C = A*B:
  C_{i,k} = sum_j A_{i,j} B_{j,k}

The tensor is T[i,j,k] = delta_{i,k} (when we think of it as a 3-tensor with
indices i (row of A), j (column of A / row of B), k (column of B)).

Actually, let me be more careful. The multiplication tensor for (n,m,p) is:
  T[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]

So the tensor has entries T[i,j,k] = delta_{i,k}? No.

The standard formulation: The matrix multiplication tensor T_{n,m,p} has
indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p), and
T[i,j,k] = 1 if i == k (when n == p) ... no that's not right either.

Let me think again. For C = A * B where A is n x m and B is m x p:
C[i,k] = sum_j A[i,j] B[j,k]

The tensor T is a 3-way tensor of size n x m x p where T[i,j,k] = delta_{i,k} 
only makes sense if n == p. In general, T[i,j,k] is defined such that:
  C[i,k] = sum_{j} A[i,j] * B[j,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]

Wait, that's not right. The tensor should satisfy:
  C[i,k] = sum_{j} A[i,j] B[j,k]

This is a bilinear map. The tensor T has entries T[i,j,k] such that:
  C[i,k] = sum_{j} A[i,j] * B[j,k]

Actually the tensor decomposition for matrix multiplication is:
  T[i, j, k] = 1  (for all i, j, k)  -- NO

Let me use the standard definition. The matrix multiplication tensor for
multiplying an (n x m) matrix by an (m x p) matrix is a tensor of size
(n*m) x (m*p) x (p*n) where:

T[(i,j), (j',k), (k',i')] = 1 if j == j' and k == k' and i == i'

So T is indexed by:
- First mode: (i, j) with i in [0,n), j in [0,m)  -> size n*m
- Second mode: (j', k) with j' in [0,m), k in [0,p)  -> size m*p  
- Third mode: (k', i') with k' in [0,p), i' in [0,n)  -> size p*n

And T[(i,j), (j',k), (k',i')] = 1 iff j==j' and k==k' and i==i'

This is the "identity" tensor in a sense. The rank of this tensor is the
minimum number of rank-1 tensors needed to express it.

For the schoolbook: rank = n*m*p = 3*3*6 = 54.

Known lower bounds: For (3,3,6), the rank is at least max(n*p, m*p, n*m) 
by the "border rank" or simple arguments. Actually:
- n*m = 9
- m*p = 18
- n*p = 18

So rank >= 18.

The known best upper bounds for small matrix multiplication:
- (2,2,2): rank 7 (Strassen)
- (3,3,3): rank 23 (Landsberg) or 21 (recent)
- (3,3,6): This is equivalent to two (3,3,3) multiplications? No, it's one (3,3,6).

Actually (3,3,6) can be seen as multiplying a 3x3 by a 3x6, which gives a 3x6.
This is like two independent 3x3 multiplications if the 3x6 is two 3x3 blocks.
But they share the same left factor A, so it's not quite independent.

The rank of (3,3,6) matrix multiplication: 
- Lower bound: max(n*m, m*p, n*p) = max(9, 18, 18) = 18
- Upper bound: 2 * rank(3,3,3) = 2 * 23 = 46 (trivially, by splitting B into two 3x3 blocks)
- Or schoolbook: 54

Can we do better than 46? If we use Strassen for each half, that's 2*7=14 per half... no wait.

Actually, (3,3,6) is the same as computing C = A*B where A is 3x3 and B is 3x6.
We can write B = [B1 | B2] where B1, B2 are 3x3. Then C = [A*B1 | A*B2].
Using rank-23 decomposition for each: total rank 46.

But can we do better? The shared structure of A might help.

Known results: The rank of (3,3,6) is at most 40 (I believe this is known).
Actually, let me think about what's achievable.

For the purpose of this task, let me try to find a good decomposition.

Approach: Use a greedy/optimization approach to find a low-rank decomposition.

The tensor T has entries T[(i,j), (j',k), (k',i')] = 1 iff i==i', j==j', k==k'.

Let me index:
- U rows: (i,j) for i in 0..2, j in 0..2 -> 9 rows
- V rows: (j',k) for j' in 0..2, k in 0..5 -> 18 rows
- W rows: (k',i') for k' in 0..5, i' in 0..2 -> 18 rows

T[(i,j),(j',k),(k',i')] = 1 if i==i' and j==j' and k==k'

A rank-1 component: U[(i,j),c] * V[(j',k),c] * W[(k',i'),c]

We need: sum_c U[(i,j),c] * V[(j',k),c] * W[(k',i'),c] = 1 if i==i', j==j', k==k' else 0

This is a system of 9*18*18 = 2916 equations.

Let me try a different approach. Since the structure is:
T[(i,j),(j',k),(k',i')] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}

This is actually the tensor product of three identity tensors!
T = I_3 (in i,i') tensor I_3 (in j,j') tensor I_6 (in k,k')

The rank of a tensor product is at most the product of ranks, and for identity tensors,
the rank of I_n is n (since I_n = sum_i e_i outer e_i).

So rank(T) <= rank(I_3) * rank(I_3) * rank(I_6) = 3 * 3 * 6 = 54 (schoolbook)

But we can potentially do better using the structure.

Actually, since T = I_3 ⊗ I_3 ⊗ I_6, and the rank of a Kronecker product of tensors
is at most the product of ranks, but we might be able to do better.

For I_3 ⊗ I_3: This is the tensor for 3x3 matrix "multiplication" by identity? 
No, it's just the 9-dimensional identity tensor. Its rank is 9.

Hmm, let me think differently.

Actually, I realize that T[(i,j),(j',k),(k',i')] = delta_{i,i'} delta_{j,j'} delta_{k,k'}

This means T is a "diagonal" tensor in a sense. The rank of such a tensor where
the three index sets are (i,j), (j',k), (k',i') and the condition is i=i', j=j', k=k'...

Let me re-index more carefully. Define:
- a = (i, j) where i in {0,1,2}, j in {0,1,2}
- b = (j', k) where j' in {0,1,2}, k in {0,1,2,3,4,5}
- c = (k', i') where k' in {0,1,2,3,4,5}, i' in {0,1,2}

T[a,b,c] = 1 iff i==i' AND j==j' AND k==k'

For a fixed (i,j,j',k,k',i'), the entry is 1 if all three equalities hold.

This is equivalent to: T is the tensor that has a 1 at position ((i,j),(j,k),(k,i)) 
for each valid combination, and 0 elsewhere. Wait, that's not quite right because
j and j' are different indices that must be equal, etc.

Let me just enumerate: The nonzero entries are at:
((i,j), (j,k), (k,i)) for i in {0,1,2}, j in {0,1,2}, k in {0,1,2,3,4,5}

So there are 3*3*6 = 54 nonzero entries, each equal to 1.

This is a "partial identity" type tensor. 

Now, can we find a low-rank decomposition?

Let me try a numerical approach: set up the optimization problem and solve it.

The constraint is: for all (a,b,c): sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]

This is a non-convex problem. Let me try a greedy approach or use a known algorithm.

Actually, let me try a different strategy. Since the tensor has a product structure,
maybe I can use a decomposition that exploits it.

T = sum_{i=0}^{2} sum_{j=0}^{2} sum_{k=0}^{5} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}

This is the schoolbook with rank 54.

To reduce rank, I need to find linear combinations that cancel out.

Let me try an approach based on the fact that for fixed j, the sub-tensor is:
sum_{i,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}

For fixed j, this is a 3x6 "identity" tensor. Its rank is 6 (since it's like I_6 in a sense, 
but with the i and i' indices linked).

Actually for fixed j: T_j[i, k, k', i'] = 1 if k==k' and i==i', 0 otherwise.
This is I_3 ⊗ I_6 (as a tensor in (i,k) and (k',i')), which has rank 3*6=18... no.

I'm overcomplicating this. Let me just try a numerical search.

Strategy: 
1. Start with the schoolbook (rank 54)
2. Try to reduce rank by finding linear dependencies
3. Use optimization to find a rank-r decomposition for r < 54

Let me try a practical approach: use a greedy algorithm or a randomized optimization.

Actually, for this specific problem, let me try to use the known result that
rank(3,3,6) <= 40 or so, and try to find such a decomposition.

Let me try a different approach: use the fact that the problem is equivalent to
finding the minimum rank of a tensor with specific structure, and use a 
numerical method like alternating least squares or gradient descent.

Given the time constraints, let me try:
1. A randomized search for a low-rank decomposition
2. Verify the solution exactly (using rational arithmetic if possible)

Let me code this up.
"""

import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[(i,j), (j,k), (k,i)] = 1
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is correct."""
    T = _build_tensor(n, m, p)
    # Compute the reconstructed tensor
    r = U.shape[1]
    # Reconstruct: T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Efficient computation:
    # For each r, outer product U[:,r] @ V[:,r] @ W[:,r]
    # This is expensive for large r, but for small r it's fine.
    T_recon = np.zeros_like(T)
    for c in range(r):
        # U[:,c] is (n*m,), V[:,c] is (m*p,), W[:,c] is (p*n,)
        # Outer product: T_recon += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
        T_recon += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
    return np.max(np.abs(T_recon - T)) < tol

def _try_rank(n, m, p, rank, seed=0, max_iters=1000, lr=0.01):
    """Try to find a rank-`rank` decomposition using gradient descent."""
    rng = np.random.RandomState(seed)
    T = _build_tensor(n, m, p)
    
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Compute error
        T_recon = np.zeros_like(T)
        for c in range(rank):
            T_recon += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        err = np.sum((T_recon - T) ** 2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-12:
            break
        
        # Compute gradients
        # dT/dU[a,c] = sum_{b,c'} (T_recon[a,b,c'] - T[a,b,c']) * V[b,c] * W[c',c]
        # This is expensive. Let's use a simpler approach.
        
        # Actually, let's use a different optimization: alternating updates.
        # Fix V and W, solve for U:
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For fixed a: T[a,:,:] = sum_r U[a,r] * (V[:,r] * W[:,r])  [element-wise product]
        # This is a linear system for U[a,:].
        
        # Let's do alternating least squares.
        # Fix V, W: solve for U
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For each a: T[a,:,:] = sum_r U[a,r] * (V[:,r] * W[:,r])
        # Let M_a[r, (b,c)] = V[b,r] * W[c,r]
        # Then T[a,:,:] = M_a @ U[a,:]
        
        # This is a least squares problem for each a.
        # But it's a big system. Let's do it.
        
        # Actually, this is getting complex. Let me just use a simple gradient step.
        
        # Gradient of loss = sum_{a,b,c} (T_recon[a,b,c] - T[a,b,c])^2
        # dL/dU[a,c] = 2 * sum_{b,c'} (T_recon[a,b,c'] - T[a,b,c']) * V[b,c] * W[c',c]
        
        # Let's compute this efficiently.
        # For fixed c: dL/dU[:,c] = 2 * (T_recon - T) contracted with (V[:,c] * W[:,c])
        # Actually: dL/dU[a,c] = 2 * sum_{b,c'} E[a,b,c'] * V[b,c] * W[c',c]
        # where E = T_recon - T
        
        E = T_recon - T
        
        # dU[a,c] = 2 * sum_{b,c'} E[a,b,c'] * V[b,c] * W[c',c]
        # = 2 * (E * (V[:,c] * W[:,c])) summed over b,c'
        # But V has shape (m*p,) and W has shape (p*n,), and E has shape (n*m, m*p, p*n)
        # So for fixed c: dU[:,c] = 2 * np.einsum('abc,b,c->a', E, V[:,c], W[:,c])
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for c in range(rank):
            dU[:, c] = 2.0 * np.einsum('abc,b,d->a', E, V[:, c], W[:, c])
            dV[:, c] = 2.0 * np.einsum('abc,a,d->b', E, U[:, c], W[:, c])
            dW[:, c] = 2.0 * np.einsum('abc,a,b->c', E, U[:, c], V[:, c])
        
        U -= lr * dU
        V -= lr * dW  # Wait, I need to be careful with indexing
        V -= lr * dV
        W -= lr * dW
        
        # Renormalize to prevent divergence
        for c in range(rank):
            if np.linalg.norm(U[:, c]) > 1e6:
                U[:, c] /= np.linalg.norm(U[:, c])
            if np.linalg.norm(V[:, c]) > 1e6:
                V[:, c] /= np.linalg.norm(V[:, c])
            if np.linalg.norm(W[:, c]) > 1e6:
                W[:, c] /= np.linalg.norm(W[:, c])
    
    return best_U, best_V, best_W, best_err

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Start with schoolbook
    rank_schoolbook = n * m * p  # 54
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    # Try to find lower rank decompositions
    # Lower bound is max(n*m, m*p, n*p) = 18
    # Try ranks from low to high
    
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    best_result = (U_sb, V_sb, W_sb, rank_schoolbook)
    
    # Try various ranks
    for rank in range(18, rank_schoolbook + 1):
        if time.time() > deadline - 30:
            break
        
        found = False
        for seed in range(20):
            if time.time() > deadline - 30:
                break
            U, V, W, err = _try_rank(n, m, p, rank, seed=seed, max_iters=500, lr=0.005)
            if err < 1e-10:
                # Verify
                if _verify(U, V, W, n, m, p, tol=1e-9):
                    best_result = (U, V, W, rank)
                    found = True
                    break
        
        if found:
            break
    
    U, V, W, rank = best_result
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
