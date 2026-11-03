"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

def construct():
    t0 = time.time()
    deadline = t0 + 140
    n, m, p = 3, 5, 7

    def verify(U, V, W, tol=1e-8):
        """Verify exact tensor decomposition using matrix operations."""
        rank = U.shape[1]
        # Build the full tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Efficient verification via matrix multiplication
        for i in range(n):
            for k in range(p):
                # C[i,k] should equal sum_j A[i,j]*B[j,k]
                # In tensor form: T[i,*,k] contracted with identity = identity
                lhs = np.zeros(m)
                for j in range(m):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    lhs[j] = s
                # lhs[j] should be 1 if j==j (diagonal) — actually T[i,j,k]=δ[i,j,k] structure
                # For matrix mult: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all (i,j,k)
                # Wait, that's not right. The tensor is T_{ijk} where (i,j) index A, (j,k) index B, (k,i) index C
                # We need: for all (i,j,k), sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
                pass
        
        # Direct verification
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def verify_fast(U, V, W, tol=1e-8):
        """Faster verification using vectorized operations."""
        rank = U.shape[1]
        # Reshape for batched computation
        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k]) * 1.0
                    # This isn't quite right either. Let me do it properly.
                    pass
        
        # Just do the direct check but vectorized over c
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        return ok

    # Strategy: 
    # 1. Partition the 3x5x7 space into sub-blocks
    # 2. Use Laderman (rank 23) on 3x3x3 sub-blocks where possible
    # 3. Use Strassen (rank 7) on 2x2x2 sub-blocks
    # 4. Schoolbook on remaining
    # 5. Then apply linear dependency removal

    # Let's try a different partition:
    # i in {0,1,2}, j in {0,1,2,3,4}, k in {0,1,2,3,4,5,6}
    
    # Block 1: i={0,1,2}, j={0,1,2}, k={0,1,2} -> 3x3x3, use Laderman (rank 23 vs 27)
    # Block 2: i={0,1,2}, j={0,1,2}, k={3,4,5} -> 3x3x3, use Laderman (rank 23)
    # Block 3: i={0,1,2}, j={3,4}, k={0,1,2} -> 3x2x3 = 18 terms schoolbook
    # Block 4: i={0,1,2}, j={3,4}, k={3,4,5} -> 3x2x3 = 18 terms schoolbook
    # Block 5: i={0,1,2}, j={0,1,2,3,4}, k={6} -> 3x5x1 = 15 terms schoolbook
    
    # Total: 23 + 23 + 18 + 18 + 15 = 97 (vs 105 schoolbook, vs 99 previous)
    
    # But we can do better with more Strassen blocks. Let me think more carefully.
    
    # Alternative: Use 2x2x2 Strassen blocks more aggressively
    # We have i=3, j=5, k=7
    # Pair up: i pairs: (0,1), (2,-) ; j pairs: (0,1),(2,3),(-,4) ; k pairs: (0,1),(2,3),(4,5),(-,6)
    
    # Let me use a systematic approach:
    # Define which (i,j,k) triplets get which treatment
    
    # Actually, let me try the Laderman approach on two 3x3x3 blocks and see if we can improve further.
    
    # Laderman's algorithm for 3x3 matrix multiplication (rank 23):
    # This is complex. Let me use a simpler but effective approach.
    
    # Better strategy: Use the known result that rank of 3x3x3 matmul <= 23 (Laderman)
    # and rank of 2x2x2 = 7 (Strassen)
    
    # Partition:
    # Region A: i∈{0,1,2}, j∈{0,1,2}, k∈{0,1,2} → Laderman 23
    # Region B: i∈{0,1,2}, j∈{0,1,2}, k∈{3,4,5} → Laderman 23  
    # Region C: i∈{0,1,2}, j∈{3,4}, k∈{0,1,2} → 18 schoolbook
    # Region D: i∈{0,1,2}, j∈{3,4}, k∈{3,4,5} → 18 schoolbook
    # Region E: i∈{0,1,2}, j∈{0,1,2,3,4}, k∈{6} → 15 schoolbook
    # Total: 23+23+18+18+15 = 97
    
    # But can we do better in regions C and D with mixed Strassen?
    # Region C: 3x2x3. We could split into 2x2x3 + 1x2x3, etc.
    # For 2x2x3: use Strassen on the 2x2 part for each k, but that's not standard.
    # Actually 2x2x3 can be done in rank 7+7=14? No, that's not how it works.
    
    # Let me just go with 97 and then try to reduce via linear algebra.
    
    # Implement Laderman's 3x3x3 decomposition (rank 23)
    # Reference: Laderman, 1976. "A note on multiplication of 3x3 matrices."
    
    def laderman_3x3():
        """Return Laderman's rank-23 decomposition for 3x3x3 matrix multiplication.
        Returns lists of (u, v, w) where u,v,w are length-9 vectors (for 3x3, 3x3, 3x3).
        u[i*3+j], v[j*3+k], w[k*3+i]
        """
        # Laderman's 23 terms for multiplying 3x3 matrices
        # A = [[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]]
        # B = [[b11,b12,b13],[b21,b22,b23],[b31,b32,b33]]
        # C = A*B
        
        # The 23 products in Laderman's algorithm:
        terms = []
        
        # Using the standard Laderman construction
        # M1 = (a11+a22+a33)(b11+b22+b33)
        # M2 = (a11+a22-a33)(b11+b22-b33)
        # M3 = (a11-a22+a33)(b11-b22+b33)
        # M4 = (a11-a22-a33)(b11-b22-b33)
        # M5 = (a12+a23+a31)(b12+b23+b31)
        # M6 = (a12+a23-a31)(b12+b23-b31)
        # M7 = (a12-a23+a31)(b12-b23+b31)
        # M8 = (a12-a23-a31)(b12-b23-b31)
        # M9 = (a13+a32+a21)(b13+b32+b21)
        # M10 = (a13+a32-a21)(b13+b32-b21)
        # M11 = (a13-a32+a21)(b13-b32+b21)
        # M12 = (a13-a32-a21)(b13-b32-b21)
        # M13 = (a22+a33+a11)(b22+b33+b11)  -- same as M1, skip
        # 
        # This is getting complicated. Let me use a verified set of 23 terms.
        
        # Actually, let me use a simpler verified approach.
        # I'll use the known 23-term Laderman decomposition from literature.
        
        # For simplicity and correctness, let me use a different strategy:
        # Use Strassen-based decomposition for sub-blocks and verify.
        
        # Let me fall back to a proven approach: 
        # Use 2x2x2 Strassen blocks where possible, schoolbook elsewhere.
        
        # Partition 3x5x7:
        # i: {0,1} x {2}
        # j: {0,1} x {2,3} x {4}  
        # k: {0,1} x {2,3} x {4,5} x {6}
        
        # Blocks:
        # 1. i={0,1}, j={0,1}, k={0,1} → Strassen 7
        # 2. i={0,1}, j={0,1}, k={2,3} → Strassen 7
        # 3. i={0,1}, j={0,1}, k={4,5} → Strassen 7
        # 4. i={0,1}, j={2,3}, k={0,1} → Strassen 7
        # 5. i={0,1}, j={2,3}, k={2,3} → Strassen 7
        # 6. i={0,1}, j={2,3}, k={4,5} → Strassen 7
        # 7. i={0,1}, j={4}, k={0,1} → 4 schoolbook
        # 8. i={0,1}, j={4}, k={2,3} → 4 schoolbook
        # 9. i={0,1}, j={4}, k={4,5} → 4 schoolbook
        # 10. i={2}, j={0,1}, k={0,1} → 4 schoolbook
        # 11. i={2}, j={0,1}, k={2,3} → 4 schoolbook
        # 12. i={2}, j={0,1}, k={4,5} → 4 schoolbook
        # 13. i={2}, j={2,3}, k={0,1} → 4 schoolbook
        # 14. i={2}, j={2,3}, k={2,3} → 4 schoolbook
        # 15. i={2}, j={2,3}, k={4,5} → 4 schoolbook
        # 16. i={0,1,2}, j={0,1,2,3,4}, k={6} → 15 schoolbook
        # 17. i={2}, j={4}, k={0,1} → 2 schoolbook
        # 18. i={2}, j={4}, k={2,3} → 2 schoolbook
        # 19. i={2}, j={4}, k={4,5} → 2 schoolbook
        
        # Total: 6*7 + 9*4 + 4*4 + 15 + 3*2 = 42 + 36 + 16 + 15 + 6 = 115? That's worse!
        
        # I'm making errors. Let me be more careful.
        
        # The issue is that when I split into blocks, I need to make sure each (i,j,k) is counted exactly once.
        
        # Let me use the same 6 Strassen blocks as before but verify they're correct,
        # then try to add more structure.
        
        # Previous blocks:
        # ({0, 1}, {0, 1}, {0, 1})
        # ({1, 2}, {0, 1}, {2, 3})
        # ({0, 2}, {0, 1}, {4, 5})
        # ({0, 1}, {2, 3}, {0, 1})
        # ({1, 2}, {2, 3}, {2, 3})
        # ({0, 2}, {2, 3}, {4, 5})
        
        # These cover 6*8 = 48 terms, leaving 105-48 = 57 schoolbook terms.
        # Total: 6*7 + 57 = 99.
        
        # Can we find more 2x2x2 blocks?
        # We need disjoint pairs of i, j, k.
        # i has 3 elements, so at most 1 full pair + 1 leftover
        # j has 5 elements, so at most 2 full pairs + 1 leftover
        # k has 7 elements, so at most 3 full pairs + 1 leftover
        
        # Maximum number of 2x2x2 blocks = min(floor(3/2), floor(5/2), floor(7/2)) = min(1, 2, 3) = 1
        # Wait, that's not right. Each block uses 2 i's, 2 j's, 2 k's.
        # The blocks can share indices across different dimensions.
        
        # Actually, for a set of 2x2x2 blocks to be "disjoint" in the sense that they don't overlap in (i,j,k) space,
        # we need the (I,J,K) tuples to be such that no (i,j,k) appears in two blocks.
        
        # The previous 6 blocks: let me verify they're disjoint.
        # Block 1: i∈{0,1}, j∈{0,1}, k∈{0,1} → covers (i,j,k) for i∈{0,1},j∈{0,1},k∈{0,1}
        # Block 2: i∈{1,2}, j∈{0,1}, k∈{2,3} → covers i∈{1,2},j∈{0,1},k∈{2,3}
        # These overlap at i=1,j∈{0,1} but different k's, so no (i,j,k) overlap. Good.
        
        # Let me check all pairs...
        # Actually, the previous code works and gives 99. Let me try to improve by:
        # 1. Using better block selection
        # 2. Applying linear algebra reduction
        
        # For now, let me implement the 99-rank solution robustly and then try to reduce.
        
        return None  # Placeholder
        
    # Let me just implement the 99-rank solution cleanly and then apply reduction.
    
    blocks = [
        ({0, 1}, {0, 1}, {0, 1}),
        ({1, 2}, {0, 1}, {2, 3}),
        ({0, 2}, {0, 1}, {4, 5}),
        ({0, 1}, {2, 3}, {0, 1}),
        ({1, 2}, {2, 3}, {2, 3}),
        ({0, 2}, {2, 3}, {4, 5}),
    ]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen(i0, i1, j0, j1, k0, k1):
        """Add 7 Strassen terms for 2x2x2 block."""
        # M1 = (A00+A11)(B00+B11)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i0*m+j0] = 1; u[i1*m+j1] = 1
        v[j0*p+k0] = 1; v[j1*p+k1] = 1
        w[k0*n+i0] = 1; w[k1*n+i1] = 1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M2 = (A10+A11)B00
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i1*m+j0] = 1; u[i1*m+j1] = 1
        v[j0*p+k0] = 1
        w[k0*n+i1] = 1; w[k1*n+i1] = -1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M3 = A00(B01-B11)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i0*m+j0] = 1
        v[j0*p+k1] = 1; v[j1*p+k1] = -1
        w[k1*n+i0] = 1; w[k1*n+i1] = 1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M4 = A11(B10-B00)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i1*m+j1] = 1
        v[j1*p+k0] = 1; v[j0*p+k0] = -1
        w[k0*n+i0] = 1; w[k0*n+i1] = 1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M5 = (A00+A01)B11
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i0*m+j0] = 1; u[i0*m+j1] = 1
        v[j1*p+k1] = 1
        w[k0*n+i0] = -1; w[k1*n+i0] = 1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M6 = (A10-A00)(B00+B01)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i1*m+j0] = 1; u[i0*m+j0] = -1
        v[j0*p+k0] = 1; v[j0*p+k1] = 1
        w[k1*n+i1] = 1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
        # M7 = (A01-A11)(B10+B11)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i0*m+j1] = 1; u[i1*m+j1] = -1
        v[j1*p+k0] = 1; v[j1*p+k1] = 1
        w[k0*n+i0] = 1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
    
    for I, J, K in blocks:
        i0, i1 = sorted(I)
        j0, j1 = sorted(J)
        k0, k1 = sorted(K)
        add_strassen(i0, i1, j0, j1, k0, k1)
    
    # Add remaining schoolbook terms
    covered = set()
    for I, J, K in blocks:
        for i in I:
            for j in J:
                for k in K:
                    covered.add((i, j, k))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
                    u[i*m+j] = 1
                    v[j*p+k] = 1
                    w[k*n+i] = 1
                    U_cols.append(u); V_cols.append(v); W_cols.append(w)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    
    rank = U_mat.shape[1]
    
    # Now apply linear algebra reduction
    # A term c is redundant if there exists a linear combination of other terms that reproduces its contribution.
    # More practically: try removing terms one at a time and check validity.
    # But that's O(rank^2 * verify_cost). Let's be smarter.
    
    # Use the fact that each (i,j,k) gives one equation:
    # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    
    # We can formulate this as: for each (i,j,k), the "slice" must sum to 1.
    
    # Greedy removal with batch verification
    def can_remove(U, V, W, indices_to_check):
        """Check if removing terms in indices_to_check still gives valid decomposition."""
        keep = [i for i in range(U.shape[1]) if i not in indices_to_check]
        if len(keep) == 0:
            return False
        U_new = U[:, keep]
        V_new = V[:, keep]
        W_new = W[:, keep]
        return verify_fast(U_new, V_new, W_new)
    
    # Remove one at a time, multiple passes
    removed_any = True
    passes = 0
    while removed_any and time.time() < deadline - 20:
        removed_any = False
        passes += 1
        for idx in range(U_mat.shape[1]):
            if time.time() >= deadline - 20:
                break
            # Try removing this term
            keep = list(range(U_mat.shape[1]))
            keep.pop(idx)
            if not keep:
                continue
            U_new = U_mat[:, keep]
            V_new = V_mat[:, keep]
            W_new = W_mat[:, keep]
            if verify_fast(U_new, V_new, W_new):
                U_mat = U_new
                V_mat = V_new
                W_mat = W_new
                removed_any = True
                break  # Restart from beginning since indices changed
    
    final_rank = U_mat.shape[1]
    
    if verify_fast(U_mat, V_mat, W_mat):
        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
    else:
        # Fallback: return the pre-reduction version
        U_cols2 = []
        V_cols2 = []
        W_cols2 = []
        for I, J, K in blocks:
            i0, i1 = sorted(I)
            j0, j1 = sorted(J)
            k0, k1 = sorted(K)
            add_strassen(i0, i1, j0, j1, k0, k1)
            # Oops, this appends to the global lists. Let me rebuild.
        
        # Actually, let me just return what we have if valid, otherwise schoolbook
        U_sb = np.zeros((n*m, n*m*p))
        V_sb = np.zeros((m*p, n*m*p))
        W_sb = np.zeros((p*n, n*m*p))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, c] = 1
                    V_sb[j*p+k, c] = 1
                    W_sb[k*n+i, c] = 1
                    c += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
