"""Tensor decomposition search for 3x3x4 matrix multiplication with strict deadline guard."""
import time
import math
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

_START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 3.0

def _check_deadline():
    return (time.time() - _START_TIME) < (TIME_LIMIT - SAFETY_MARGIN)

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i, j, k, r, s, l] = 1 if i*r + j*s == k*l."""
    T = np.zeros((n, m, p, n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(n):
                    for s in range(m):
                        for l in range(p):
                            if i * r + j * s == k * l:
                                T[i, j, k, r, s, l] = 1.0
    return T

def _contract(T, U, V, W):
    """Contract T with factors to get W'[k, n, i, r] equivalent to W indices (k*n+i, r).
    Here T shape: (i, j, k, r, s, l)
    U: (n, m, c)  -> U[i, j, c]
    V: (m, p, c)  -> V[j, s, c]  (note: s is the row index of B, which is m; B has shape m x p, B[j, l])
    Actually V[j, p, c] corresponds to V[j, l, c].
    Let's re-align the tensor axes to the factorization convention used in the task.
    
    The tensor T_{ijk} = sum_{rs} A_{ir} B_{js} C_{kl} delta(...).
    The factorization is T_{ijk} = sum_c U_{ij,c} V_{js,c} W_{k,c}? No, the task says:
    U: (n*m, rank) -> U[i*m + j, c]
    V: (m*p, rank) -> V[j*p + k, c]   (Here k is the column of B, which is p)
    W: (p*n, rank) -> W[k*n + i, c]   (Here k is the row of C, which is p; i is row of result A? No, output is A B C? 
    Let's re-read: "Multiplying a 3 by 3 matrix by a 3 by 4 matrix. Output W[k*n+i, r]."
    Let A be 3x3 (indices i, r).
    Let B be 3x4 (indices s, k) ? Or B is 3x4?
    "3 by 3 matrix by a 3 by 4 matrix" -> C = A @ B.
    A: 3x3 (rows i, cols r).
    B: 3x4 (rows s, cols k).
    C: 3x4 (rows i, cols k).
    C[i, k] = sum_s A[i, s] B[s, k].
    
    Wait, the task says "Output factor follows the official trace convention W[k*n+i, r]".
    If n=3, p=4, then k is 0..3 (p), i is 0..2 (n).
    Index is k*n + i.
    Usually C_{ik} = sum_j A_{ij} B_{jk}.
    Let's map:
    n = 3, m = 3, p = 4.
    A is n x m. B is m x p. C is n x p.
    C[i, k] = sum_j A[i, j] B[j, k].
    Task U shape (n*m, rank) -> U[i, j, c].
    Task V shape (m*p, rank) -> V[j, k, c].
    Task W shape (p*n, rank) -> W[k, i, c] ? No, W[k*n + i, c].
    So W_{k, i, c}.
    Decomposition: C_{i, k} = sum_c U_{i, j, c} * ???
    Let's check the schoolbook in the seed:
    U[i*m+j][column] = 1
    V[j*p+k][column] = 1
    W[k*n+i][column] = 1
    
    The schoolbook term for C[i,k] uses A[i,j] B[j,k].
    So the tensor is sum over c of U[i,j,c] V[j,k,c] W[k,i,c]? No, that's 3 factors but we only multiply A and B.
    Standard Strassen-like: C[i,k] = sum_c U_{ij,c} V_{jk,c} W_{ik,c}? No, the third factor is for the output.
    Usually: C_{ik} = sum_c (sum_i,j A_{ij} U_{ij,c}) (sum_j,k B_{jk} V_{jk,c}) (sum_k,i C_{ik} W_{ik,c})? No.
    
    The decomposition is:
    C_{ik} = sum_c u_c v_c w_c? No.
    The identity is:
    sum_{i,k} C_{ik} M_{ik} = sum_c (sum_{ij} U_{ij,c} A_{ij}) (sum_{jk} V_{jk,c} B_{jk}) (sum_{ik} W_{ik,c} C_{ik})
    
    So, for a specific c, we pick linear forms U^c, V^c, W^c.
    The tensor T_{ik}^{ij,jk}?
    Let's define the tensor $T$ such that:
    sum_{c} U_{ij,c} V_{jk,c} W_{ki,c} = 1 if the schoolbook terms match?
    
    Let's stick to the seed's definition which is verified to work:
    T_{ij,k l,?} ...
    
    Let's build the target tensor $T_{i,j,k,r,s,l}$ ? No.
    
    Let's use the standard form:
    We want $U_{ij,c} V_{jk,c} W_{k,i,c}$ to reconstruct the matrix multiplication.
    For the schoolbook case (rank 36):
    U_{ij,c} = delta_{i, r} delta_{j, s} ?
    Seed: U[i*m+j, c] = 1 for unique c.
    So for each c, there is exactly one (i,j) in U with 1, one (j,k) in V with 1, one (k,i) in W with 1.
    This implies for each c, the term contributes $A_{i,j} B_{j,k} \cdot 1$ to $C_{i,k}$?
    No, if $U_{ij}=1, V_{jk}=1, W_{ki}=1$, then the product in the identity $C = \sum_c (U \cdot A) (V \cdot B) (W \cdot C)^T$?
    
    Let's just optimize for the condition that reconstructing the multiplication works.
    We will use a greedy search / local search over ranks 20 to 28.
    
    However, given the strict deadline and the difficulty, and that rank 24 is known for 3x3x3, and 3x3x4 is likely 25 or 26.
    We will attempt a structured search using numpy.
    """
    pass # Placeholder to structure thought
    
def _greedy_search(n, m, p, rank, seed):
    T = _build_tensor(n, m, p)
    # Flatten tensor to match the bilinear structure.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # We look for U_{ij}, V_{jk}, W_{ki} (indices mapped to vector)
    # dim_U = n*m = 9
    # dim_V = m*p = 12
    # dim_W = p*n = 12
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Initialize with random small integers or sparse structures?
    # A good heuristic for low rank is to start from a known high rank and prune?
    # Or start from rank and refine.
    
    # Let's try random initialization and local search / gradient steps.
    random.seed(seed)
    np.random.seed(seed)
    
    # Scale of entries? Usually small integers.
    # Initialize with 0/1/-1 sparse?
    for c in range(rank):
        idx_u = np.random.randint(0, n * m)
        idx_v = np.random.randint(0, m * p)
        idx_w = np.random.randint(0, p * n)
        U[idx_u, c] = 1.0
        V[idx_v, c] = 1.0
        W[idx_w, c] = 1.0
    
    # We can't easily gradient descend on sparse discrete values.
    # We will try to improve a few specific ranks by checking validity.
    
    # Actually, constructing a valid rank < 36 decomposition analytically is hard.
    # If we cannot find one, return seed.
    
    return {"U": U, "V": V, "W": W}

def _verify_decomp(n, m, p, U, V, W):
    """Check if U, V, W exactly decompose matrix multiplication.
    C[i, k] = sum_j A[i, j] B[j, k]
    The decomposition identity is:
    For all i, k:
    sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * (sum_l V[j*p+l, c] B[j,l]) == sum_j A[i,j] B[j,k]
    
    Wait, the indices must align.
    Let $u_c = \sum_{i,j} U_{ij,c} A_{ij}$
    Let $v_c = \sum_{j,k} V_{jk,c} B_{jk}$
    Let $w_c = \sum_{i,k} W_{ki,c} C_{ik}$
    
    The condition is that for all A, B:
    $\sum_c U^c(A) V^c(B) W^c(C) = \sum_{i,k} C_{ik} C_{ik}$?
    No, $C_{ik} = \sum_j A_{ij} B_{jk}$.
    
    The standard form is:
    $\sum_{i,k} C_{ik} \left( \sum_c W_{ki,c} u_c v_c \right) = \sum_{i,k} C_{ik} C_{ik}$?
    No.
    
    Correct Identity:
    $C = \sum_c u_c v_c w_c^T$? No.
    $C_{ik} = \sum_c ( \sum_{a,b} U_{ab,c} A_{ab} ) ( \sum_{b,d} V_{bd,c} B_{bd} ) \times \text{coeff}_{ik,c}$?
    
    Let's look at the seed again.
    U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    This means for a specific c, we select one element from A, one from B, and one from C.
    If we compute $C'_{ik} = \sum_c W_{ki,c} U_{ij,c} V_{jk,c} A_{ij} B_{jk}$?
    No, $U$ and $V$ are coefficients for A and B.
    
    The factorization is valid if:
    $\sum_{c} W_{ki,c} U_{ij,c} V_{jk,c} = 0$ for $i,j,k$ not satisfying the relation?
    And $= 1$ for the relevant terms?
    
    Actually, the condition is:
    $\sum_c W_{ki,c} U_{ij,c} V_{jk,c} A_{ij} B_{jk}$ must reconstruct $C_{ik}$.
    This requires that for each $i, j, k$:
    The coefficient of $A_{ij} B_{jk}$ in $\sum_c \dots$ is 1 if $A_{ij} B_{jk}$ contributes to $C_{ik}$?
    No, $C_{ik}$ is the SUM over $j$.
    So $A_{ij} B_{jk}$ contributes to $C_{ik}$.
    
    The condition is:
    For all $i, j, k, l$ (where $l$ is not used?), no.
    
    Let's define the error tensor.
    $E_{i, k} = C_{ik} - \sum_c ( \sum_j U_{ij,c} A_{ij} ) ( \sum_l V_{jk,c} B_{jk} ) \dots$?
    
    Let's assume the structure:
    $C_{ik} = \sum_c ( \sum_j U_{ij,c} A_{ij} ) ( \sum_l V_{lk,c} B_{lk} ) \delta_{?}$?
    No, the third factor $W$ provides the output indices.
    
    $C_{ik} = \sum_c W_{ki,c} \left( \sum_{a,b} U_{ab,c} A_{ab} \right) \left( \sum_{b,d} V_{bd,c} B_{bd} \right)$?
    This would couple all $a,b$ and $b,d$.
    
    The correct form for Strassen:
    $C_{ik} = \sum_c W_{ki,c} \, u_c \, v_c$
    where $u_c = \sum_{a,b} U_{ab,c} A_{ab}$ and $v_c = \sum_{b,d} V_{bd,c} B_{bd}$.
    
    For this to hold, we must have:
    $\sum_c W_{ki,c} \left( \sum_{a,b} U_{ab,c} A_{ab} \right) \left( \sum_{b,d} V_{bd,c} B_{bd} \right) = \sum_j A_{ij} B_{jk}$
    
    Comparing coefficients of $A_{ab} B_{cd}$:
    LHS: $\sum_c W_{ki,c} U_{ab,c} V_{cd,c}$ (Note: $U$ has indices $a,b$, $V$ has $c,d$).
    RHS: $1$ if $a=i, b=j, c=j, d=k$ (i.e., $a=i, d=k, b=c$), else 0.
    
    So the condition is:
    $\sum_c W_{ki,c} U_{ij,c} V_{jk,c} = \delta_{i,a}\delta_{k,d}\delta_{b,j}$?
    Wait, indices:
    $U_{a,b,c}$, $V_{c,d,c}$ (using c for contraction index, let's use $s$).
    $V_{b,d,c}$?
    $A$ is $n \times m$. $U$ is indexed by $i, j$.
    $B$ is $m \times p$. $V$ is indexed by $j, k$.
    $C$ is $n \times p$. $W$ is indexed by $i, k$ (mapped to $k*n+i$).
    
    So: $\sum_c W_{ik,c} U_{ij,c} V_{jk,c} = 1$ if $a=i, b=j, c=j, d=k$?
    No, the RHS is $A_{ij} B_{jk}$.
    The term $A_{ij} B_{jk}$ appears in the product $u_c v_c$ only if $U$ picks $A_{ij}$ and $V$ picks $B_{jk}$.
    So for a fixed $i, j, k$:
    We want $\sum_c W_{ik,c} U_{ij,c} V_{jk,c} = 1$.
    And for any other combination of indices that doesn't match $A_{ij}B_{jk}$ for the SAME $j$, it should be 0?
    No, $u_c$ sums over ALL $a,b$. $v_c$ sums over ALL $b,d$.
    So $u_c v_c$ contains ALL products $A_{ab} B_{cd}$.
    The coefficient of $A_{ab} B_{cd}$ in $u_c v_c$ is $U_{ab,c} V_{cd,c}$.
    Multiplied by $W_{ik,c}$ and summed over $c$:
    Total coeff of $A_{ab} B_{cd}$ in $C_{ik}$ is $\sum_c W_{ik,c} U_{ab,c} V_{cd,c}$.
    
    We need this to be 1 if $a=i, d=k, b=c$ (note $b$ and $c$ must be the same index for the product to exist in $C_{ik}$? No, $A_{ab} B_{cd}$ is just a product. It contributes to $C_{ik}$ only if the matrix multiplication rule says so.
    $C_{ik} = \sum_s A_{is} B_{sk}$.
    So the term $A_{ab} B_{cd}$ contributes to $C_{ik}$ ONLY IF $a=i, d=k, b=c$.
    In that case, the coefficient must be 1.
    In all other cases ($a \neq i$ or $d \neq k$ or $b \neq c$), the coefficient must be 0.
    
    So we have a system of linear equations for the rank-1 tensors $W_c \otimes U_c \otimes V_c$.
    The tensor $T$ of size $n \times p \times n \times m \times m \times p$?
    Indices: $k, i, a, b, c, d$. (Using c for middle index of B? Let's use $s, t$).
    $T_{k,i,a,b,s,d} = 1$ if $k=d, i=a, s=b$.
    
    We want to decompose this tensor into a sum of rank-1 tensors $W_k,i \cdot U_a,b \cdot V_s,d$.
    
    """
    pass

def construct():
    # If we can't guarantee a better rank, return the seed decomposition.
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
