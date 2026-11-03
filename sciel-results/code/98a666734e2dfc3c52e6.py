import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Build target tensor support
    support = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                support.append((u, v, w))
    
    support_set = set(support)
    target_rank = 12

    # Fallback to schoolbook
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(nm)]
    V_sb = [[0] * schoolbook_rank for _ in range(mp)]
    W_sb = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    start_time = time.time()
    deadline = 110.0

    def compute_tensor(U, V, W):
        """Compute the full tensor from factor matrices."""
        R = U.shape[1]
        C = np.zeros((nm, mp, pn))
        for c in range(R):
            C += np.einsum('u,v,w->uvw', U[:, c], V[:, c], W[:, c])
        return C
    
    def build_target():
        T = np.zeros((nm, mp, pn))
        for (u, v, w) in support:
            T[u, v, w] = 1.0
        return T
    
    T = build_target()
    
    def residual_norm(U, V, W):
        C = compute_tensor(U, V, W)
        return np.linalg.norm(C - T)
    
    def verify_exact(U, V, W, atol=1e-8):
        """Check if U,V,W exactly decompose T with half-integer entries."""
        # Check half-integer constraint
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=atol):
                return False
        C = compute_tensor(U, V, W)
        return np.allclose(C, T, atol=atol)
    
    def snap_half(x):
        return np.round(x * 2) / 2.0

    # Strategy 1: Structured Enumeration of Pairings
    # We hypothesize that a rank-12 decomposition exists where each rank-1 component
    # covers exactly 2 support entries.
    # For a rank-1 component U_c ⊗ V_c ⊗ W_c to have non-zero entries only at (u1,v1,w1) and (u2,v2,w2),
    # the supports of U_c, V_c, W_c must have sizes such that their Cartesian product
    # intersects the support set only at those two points.
    # Given the specific geometry of matrix multiplication, the only way to have exactly 2 non-zero
    # entries in the intersection with the support (assuming no cancellation from other components for simplicity
    # or that we are constructing a "clean" decomposition) is if the indices share enough to restrict the product.
    
    # Actually, let's look at the structure.
    # If we pair (u1, v1, w1) and (u2, v2, w2).
    # If u1 == u2, v1 == v2, w1 != w2:
    #   U_c has support {u1}, V_c has support {v1}, W_c has support {w1, w2}.
    #   Then U_c[u1]*V_c[v1]*W_c[w1] = 1 and U_c[u1]*V_c[v1]*W_c[w2] = 1.
    #   This implies W_c[w1] = W_c[w2].
    #   If W_c[w1]=1, W_c[w2]=1, U_c[u1]=1, V_c[v1]=1, this works.
    #   Are there other support entries?
    #   The component contributes to (u1, v1, w1) and (u1, v1, w2).
    #   Since (u1, v1, w1) and (u1, v1, w2) are in support, this is a valid "clean" component
    #   IF NO OTHER component contributes to these entries (or if we sum them up correctly).
    #   Wait, if we sum 12 such components, and each covers 2 entries, and every entry is covered exactly once,
    #   then the sum is exactly the target tensor.
    #   This requires a perfect matching of the 24 support entries into 12 pairs, such that for each pair,
    #   the "clean" rank-1 component (with 1s at the necessary indices) does not accidentally create
    #   non-zero entries at OTHER support positions that are NOT in the pair.
    #   Actually, if the component is clean (i.e., its full support is contained in the pair's indices),
    #   it cannot create entries elsewhere.
    #   Condition for clean component:
    #   The set of non-zero entries of U_c ⊗ V_c ⊗ W_c must be a subset of {(u1,v1,w1), (u2,v2,w2)}.
    #   This implies |supp(U)| * |supp(V)| * |supp(W)| <= 2.
    #   Since we need 2 entries, and entries are distinct, we must have:
    #   Case 1: |supp(U)|=1, |supp(V)|=1, |supp(W)|=2. (u1=u2, v1=v2, w1!=w2)
    #   Case 2: |supp(U)|=1, |supp(V)|=2, |supp(W)|=1. (u1=u2, v1!=v2, w1=w2)
    #   Case 3: |supp(U)|=2, |supp(V)|=1, |supp(W)|=1. (u1!=u2, v1=v2, w1=w2)
    
    # Let's check if such a perfect matching exists.
    # We will build a graph where nodes are support entries.
    # An edge exists between two entries if they form a "clean pair" (one of the 3 cases above).
    # Then we look for a perfect matching in this graph.
    
    def can_form_clean_pair(e1, e2):
        u1, v1, w1 = e1
        u2, v2, w2 = e2
        # Case 1: u1==u2, v1==v2, w1!=w2
        if u1 == u2 and v1 == v2 and w1 != w2:
            return True
        # Case 2: u1==u2, v1!=v2, w1==w2
        if u1 == u2 and v1 != v2 and w1 == w2:
            return True
        # Case 3: u1!=u2, v1==v2, w1==w2
        if u1 != u2 and v1 == v2 and w1 == w2:
            return True
        return False

    # Build adjacency list for matching
    # Nodes are 0..23
    adj = [[] for _ in range(24)]
    for i in range(24):
        for j in range(i+1, 24):
            if can_form_clean_pair(support[i], support[j]):
                adj[i].append(j)
                adj[j].append(i)
    
    # Find perfect matching using backtracking
    # 24 nodes is small enough for backtracking with pruning
    used = [False] * 24
    matching = [-1] * 24
    
    def find_matching(idx):
        if idx == 24:
            return True
        if used[idx]:
            return find_matching(idx + 1)
        
        for neighbor in adj[idx]:
            if not used[neighbor]:
                used[idx] = True
                used[neighbor] = True
                matching[idx] = neighbor
                matching[neighbor] = idx
                
                if find_matching(idx + 1):
                    return True
                
                used[idx] = False
                used[neighbor] = False
                matching[idx] = -1
                matching[neighbor] = -1
        return False

    best_solution = None
    
    if time.time() < deadline - 5:
        # Try to find the perfect matching
        if find_matching(0):
            # We found a perfect matching. Now construct U, V, W.
            U = np.zeros((nm, target_rank))
            V = np.zeros((mp, target_rank))
            W = np.zeros((pn, target_rank))
            
            # Assign each pair to a column
            col = 0
            for i in range(24):
                if matching[i] != -1 and i < matching[i]: # Process each pair once
                    j = matching[i]
                    e1 = support[i]
                    e2 = support[j]
                    u1, v1, w1 = e1
                    u2, v2, w2 = e2
                    
                    # Determine the structure
                    # We set the values to 1.
                    # Since it's a clean pair, the product is 1*1*1=1 at both positions.
                    # And 0 elsewhere.
                    
                    if u1 == u2 and v1 == v2:
                        # Case 1: W has 2 entries
                        U[u1, col] = 1.0
                        V[v1, col] = 1.0
                        W[w1, col] = 1.0
                        W[w2, col] = 1.0
                    elif u1 == u2 and w1 == w2:
                        # Case 2: V has 2 entries
                        U[u1, col] = 1.0
                        V[v1, col] = 1.0
                        V[v2, col] = 1.0
                        W[w1, col] = 1.0
                    elif v1 == v2 and w1 == w2:
                        # Case 3: U has 2 entries
                        U[u1, col] = 1.0
                        U[u2, col] = 1.0
                        V[v1, col] = 1.0
                        W[w1, col] = 1.0
                    
                    col += 1
            
            # Verify
            if verify_exact(U, V, W):
                best_solution = {"U": U.tolist(), "V": V.tolist(), "W": Ws.tolist()} if False else {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if best_solution:
        return best_solution

    # Strategy 2: If perfect matching failed, try a more general search.
    # The "clean" assumption might be too strong. Cross-terms might cancel.
    # However, given the score is 1.0 (schoolbook), any improvement is good.
    # Let's try a simple local search / gradient descent on a relaxed problem,
    # then snap.
    
    # But the previous attempts at ALS were flawed.
    # Let's try a different angle: Randomized search for integer/half-integer solutions.
    # Since the space is small (24 entries, rank 12), we can try to solve it as a constraint satisfaction problem.
    # But that's complex.
    
    # Let's stick to the fallback if the perfect matching didn't work.
    # Why would it not work?
    # Let's analyze the support structure.
    # n=2, m=3, p=4.
    # Support entries: (i*m+j, j*p+k, k*n+i)
    
    # Let's trace the edges.
    # Entry (i,j,k) connects to (i',j',k') if they share 2 coordinates in the (u,v,w) space.
    # u = i*m+j
    # v = j*p+k
    # w = k*n+i
    
    # Two entries (i,j,k) and (i',j',k') form a clean pair if:
    # 1. u=u', v=v', w!=w'  => i=i', j=j', k!=k' (since u,v fixed implies i,j,k fixed? No.)
    #    u=u' => i*m+j = i'*m+j'.
    #    v=v' => j*p+k = j'*p+k'.
    #    If i=i', j=j', then k!=k' is possible.
    #    So (i,j,k) and (i,j,k') are a clean pair if k!=k'.
    #    This corresponds to Case 1.
    
    # 2. u=u', v!=v', w=w' => i=i', j!=j', k=k'?
    #    u=u' => i*m+j = i'*m+j'.
    #    w=w' => k*n+i = k'*n+i'.
    #    If i=i', k=k', then w is fixed.
    #    v!=v' => j*p+k != j'*p+k. Since k=k', j!=j'.
    #    So (i,j,k) and (i,j',k) are a clean pair if j!=j'.
    #    This corresponds to Case 2.
    
    # 3. u!=u', v=v', w=w' => i!=i', j=j', k=k'?
    #    v=v' => j*p+k = j'*p+k'.
    #    w=w' => k*n+i = k'*n+i'.
    #    If j=j', k=k', then v is fixed.
    #    u!=u' => i*m+j != i'*m+j. Since j=j', i!=i'.
    #    So (i,j,k) and (i',j,k) are a clean pair if i!=i'.
    #    This corresponds to Case 3.
    
    # So the graph is essentially the Cartesian product graph of the indices (i,j,k)
    # where edges connect nodes differing in exactly one index.
    # This is a grid graph!
    # Dimensions: i in {0,1}, j in {0,1,2}, k in {0,1,2,3}.
    # The graph is a 2 x 3 x 4 grid.
    # We need a perfect matching in this grid graph.
    # A grid graph with even number of nodes (24) always has a perfect matching.
    # In fact, we can just pair them up simply.
    # Pair (i,j,k) with (i,j,k+1) for k=0,2?
    # Or pair (i,j,k) with (i,j,k+1) and (i,j,k+2) with (i,j,k+3)?
    # Since p=4 is even, we can pair k's within each (i,j) slice.
    # For each i in {0,1}, j in {0,1,2}:
    #   Pair (i,j,0) with (i,j,1)
    #   Pair (i,j,2) with (i,j,3)
    # This gives 2 * 3 * 2 = 12 pairs.
    # Each pair differs only in k.
    # This corresponds to Case 1 (u=u', v=v', w!=w').
    # Wait, does (i,j,0) and (i,j,1) have u=u', v=v'?
    # u = i*m+j. Yes.
    # v = j*p+k. v1 = j*4+0, v2 = j*4+1. v1 != v2!
    # So Case 1 requires v1=v2, which implies k1=k2. But we need w1!=w2, which implies k1!=k2 (since n=2, i fixed).
    # Contradiction.
    
    # Let's re-evaluate Case 1:
    # u=u', v=v', w!=w'.
    # u = i*m+j.
    # v = j*p+k.
    # w = k*n+i.
    # If u=u' and v=v', then i=i' and j=j' (since m=3, p=4, unique representation).
    # If i=i' and j=j', then v = j*p+k. For v=v', we need k=k'.
    # If k=k', then w = k*n+i is also equal.
    # So w1=w2.
    # But Case 1 requires w1!=w2.
    # Therefore, Case 1 is IMPOSSIBLE for this specific matrix multiplication tensor!
    # Because u and v uniquely determine i,j,k.
    
    # Let's re-evaluate Case 2:
    # u=u', v!=v', w=w'.
    # u=u' => i=i', j=j'.
    # Then v = j*p+k. v!=v' => k!=k'.
    # w = k*n+i. w=w' => k*n+i = k'*n+i'.
    # Since i=i', k*n = k'*n => k=k'.
    # Contradiction with k!=k'.
    # So Case 2 is IMPOSSIBLE.
    
    # Let's re-evaluate Case 3:
    # u!=u', v=v', w=w'.
    # v=v' => j=j', k=k'.
    # w=w' => k*n+i = k'*n+i'.
    # Since k=k', i=i'.
    # Then u = i*m+j. u=u'.
    # Contradiction with u!=u'.
    # So Case 3 is IMPOSSIBLE.
    
    # Conclusion: There are NO clean pairs!
    # Every rank-1 component that covers 2 support entries must create "cross terms" at other positions.
    # Therefore, the perfect matching strategy with clean components fails.
    
    # We must rely on cancellation of cross terms.
    # This is the hard part.
    
    # Given the time constraint and the difficulty of finding a rank-12 decomposition with cancellation,
    # and the fact that the current score is 1.0 (schoolbook rank 24),
    # we should return the schoolbook decomposition.
    
    # Is it possible to get a score better than 1.0?
    # The score is seed/value. Seed is 24.
    # If we find rank 12, score is 2.0.
    # If we find rank 18, score is 24/18 = 1.33.
    # Even a small improvement is good.
    
    # However, without a valid decomposition, we must return a valid one.
    # The schoolbook decomposition is valid.
    
    return fallback
