"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
import sys

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    nm, mp, pn = n*m, m*p, p*n  # 4, 10, 10
    deadline = time.time() + 140  # safety margin before 150s kill

    # Build target tensor for verification
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def _verify(U, V, W, tol=1e-10):
        """Verify that U, V, W decompose T exactly (within tolerance)."""
        R = np.einsum('ad,bd,cd->abc', U, V, W)
        return np.allclose(R, T, atol=tol, rtol=0)

    def _build_rank18():
        """Build a verified rank-18 decomposition using Strassen + schoolbook."""
        rank = 18
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Indices:
        # U: i*m+j: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # V: j*p+k: j=0: k=0..4 -> 0,1,2,3,4; j=1: k=0..4 -> 5,6,7,8,9
        # W: k*n+i: k=0: 0,1; k=1: 2,3; k=2: 4,5; k=3: 6,7; k=4: 8,9
        
        # Strassen for A*B1a (B columns k=0,1)
        # Standard Strassen: M1..M7 for 2x2*2x2
        # c00 = M1 - M2 + M3 + M4
        # c01 = M1 + M5 - M6 - M7
        # c10 = M2 + M6 - M3 - M4
        # c11 = M1 - M5 + M7
        # where:
        # M1=(a00+a11)(b00+b11)
        # M2=(a10+a11)*b00
        # M3=a00*(b11-b10)
        # M4=a11*(b00-b10)
        # M5=(a00+a01)*b10
        # M6=(a10+a11)*(b01-b00)
        # M7=(a01-a10)*(b00+b10)
        
        # For block (k=0,1): V indices: b00=0, b01=1, b10=5, b11=6
        # W indices: c00=0, c01=2, c10=1, c11=3
        
        terms_block1 = [
            # M1: (a00+a11)(b00+b11)
            ({0:1, 3:1}, {0:1, 6:1}, {0:1, 2:1, 1:-1, 3:1}),
            # M2: (a10+a11)*b00
            ({2:1, 3:1}, {0:1}, {0:-1, 1:1, 2:-1, 3:-1}),
            # M3: a00*(b11-b10)
            ({0:1}, {6:1, 5:-1}, {0:1, 2:1, 1:-1, 3:-1}),
            # M4: a11*(b00-b10)
            ({3:1}, {0:1, 5:-1}, {0:1, 2:-1, 1:1, 3:-1}),
            # M5: (a00+a01)*b10
            ({0:1, 1:1}, {5:1}, {0:-1, 2:-1, 1:1, 3:1}),
            # M6: (a10+a11)*(b01-b00)
            ({2:1, 3:1}, {1:1, 0:-1}, {0:1, 2:-1, 1:1, 3:-1}),
            # M7: (a01-a10)*(b00+b10)
            ({1:1, 2:-1}, {0:1, 5:1}, {0:1, 2:-1, 1:-1, 3:1}),
        ]
        
        # For block (k=2,3): V indices: b02=2, b03=3, b12=7, b13=8
        # W indices: c02=4, c03=6, c12=5, c13=7
        
        terms_block2 = [
            # M1: (a00+a11)(b02+b12)
            ({0:1, 3:1}, {2:1, 7:1}, {4:1, 6:1, 5:-1, 7:1}),
            # M2: (a10+a11)*b02
            ({2:1, 3:1}, {2:1}, {4:-1, 5:1, 6:-1, 7:-1}),
            # M3: a00*(b12-b11)  -- wait, b11 here means b12 in this block context
            # Actually for block (k=2,3): b02, b03 are row 0; b12, b13 are row 1
            # M3: a00*(b13-b12)  [analogous to a00*(b11-b10)]
            ({0:1}, {8:1, 7:-1}, {4:1, 6:1, 5:-1, 7:-1}),
            # M4: a11*(b02-b12)  [analogous to a11*(b00-b10)]
            ({3:1}, {2:1, 7:-1}, {4:1, 6:-1, 5:1, 7:-1}),
            # M5: (a00+a01)*b12  [analogous to (a00+a01)*b10]
            ({0:1, 1:1}, {7:1}, {4:-1, 6:-1, 5:1, 7:1}),
            # M6: (a10+a11)*(b03-b02)  [analogous to (a10+a11)*(b01-b00)]
            ({2:1, 3:1}, {3:1, 2:-1}, {4:1, 6:-1, 5:1, 7:-1}),
            # M7: (a01-a10)*(b02+b12)  [analogous to (a01-a10)*(b00+b10)]
            ({1:1, 2:-1}, {2:1, 7:1}, {4:1, 6:-1, 5:-1, 7:1}),
        ]
        
        t = 0
        for u_vals, v_vals, w_vals in terms_block1:
            for idx, val in u_vals.items():
                U[idx, t] += val
            for idx, val in v_vals.items():
                V[idx, t] += val
            for idx, val in w_vals.items():
                W[idx, t] += val
            t += 1
        
        for u_vals, v_vals, w_vals in terms_block2:
            for idx, val in u_vals.items():
                U[idx, t] += val
            for idx, val in v_vals.items():
                V[idx, t] += val
            for idx, val in w_vals.items():
                W[idx, t] += val
            t += 1
        
        # Schoolbook for column k=4
        # c04 = a00*b04 + a01*b14
        U[0, t] = 1; V[4, t] = 1; W[8, t] = 1; t += 1
        U[1, t] = 1; V[9, t] = 1; W[8, t] = 1; t += 1
        # c14 = a10*b04 + a11*b14
        U[2, t] = 1; V[4, t] = 1; W[9, t] = 1; t += 1
        U[3, t] = 1; V[9, t] = 1; W[9, t] = 1; t += 1
        
        assert t == rank, f"Expected {rank} terms, got {t}"
        
        if _verify(U, V, W):
            return U, V, W
        return None

    def _als_rank_search(rank, U0, V0, W0, max_iter=200, tol=1e-10):
        """ALS optimization to find a rank-r decomposition starting from initial guess."""
        U = U0.copy()
        V = V0.copy()
        W = W0.copy()
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            # Update U: for each a, U[a,:] should satisfy sum_d V[b,d]*W[c,d]*U[a,d] = T[a,b,c]
            # Fix V,W: minimize ||U @ (V^T * W^T elementwise) - T||
            # Actually: T_reshaped = U @ (V ⊙ W) where ⊙ is Khatri-Rao product
            # V_kron_W[d, b*n+c] = V[b,d]*W[c,d]
            VW = np.outer(V.T, W.T).reshape(rank, mp*pn)  # shape (rank, mp*pn)
            # T_reshaped[a, b*pn+c] = T[a,b,c]
            T_flat = T.reshape(nm, mp*pn)
            # U @ VW = T_flat => U = T_flat @ VW^+ (pseudo-inverse)
            try:
                U_new = np.linalg.lstsq(VW, T_flat, rcond=None)[0].T
            except:
                break
            U = U_new
            
            # Update V
            UW = np.outer(U.T, W.T).reshape(rank, nm*pn)
            T_flat2 = T.transpose(1, 0, 2).reshape(mp, nm*pn)
            try:
                V_new = np.linalg.lstsq(UW, T_flat2, rcond=None)[0].T
            except:
                break
            V = V_new
            
            # Update W
            UV = np.outer(U.T, V.T).reshape(rank, nm*mp)
            T_flat3 = T.transpose(2, 0, 1).reshape(pn, nm*mp)
            try:
                W_new = np.linalg.lstsq(UV, T_flat3, rcond=None)[0].T
            except:
                break
            W = W_new
            
            if _verify(U, V, W, tol=tol):
                return U, V, W, True
        
        return U, V, W, _verify(U, V, W, tol=tol)

    # First, get a valid rank-18 decomposition
    r18 = _build_rank18()
    if r18 is None:
        # Fallback: schoolbook rank 20
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        for c in range(rank):
            i = c // (m*p)
            rem = c % (m*p)
            j = rem // p
            k = rem % p
            U[i*m+j, c] = 1.0
            V[j*p+k, c] = 1.0
            W[k*n+i, c] = 1.0
        return {"U": U.astype(int).tolist(), "V": V.astype(int).tolist(), "W": W.astype(int).tolist()}
    
    U18, V18, W18 = r18
    
    # Try to find rank-17 via ALS from perturbations of rank-18
    best_rank = 18
    best_decomp = (U18, V18, W18)
    
    if time.time() < deadline - 30:
        # Try ALS for rank 17
        # Strategy: take rank-18 and try to merge two terms
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Create rank-17 initial guess by merging two random pairs from rank-18
            # or by random perturbation
            np.random.seed(attempt * 42 + 7)
            
            # Method 1: drop one term and optimize
            U17 = U18[:, :17].copy()
            V17 = V18[:, :17].copy()
            W17 = W18[:, :17].copy()
            
            # Add small noise to encourage finding different solution
            U17 += np.random.randn(nm, 17) * 0.01
            V17 += np.random.randn(mp, 17) * 0.01
            W17 += np.random.randn(pn, 17) * 0.01
            
            U_r, V_r, W_r, success = _als_rank_search(17, U17, V17, W17, max_iter=100)
            if success:
                best_rank = 17
                best_decomp = (U_r, V_r, W_r)
                break
            
            # Method 2: merge two terms from rank-18
            if attempt % 2 == 0:
                idx1 = np.random.randint(0, 18)
                idx2 = np.random.randint(0, 18)
                if idx1 == idx2:
                    continue
                # Merge: new term = sum of two
                U17 = U18.copy()
                V17 = V18.copy()
                W17 = W18.copy()
                U17[:, idx2] += U17[:, idx1]
                V17[:, idx2] += V17[:, idx1]
                W17[:, idx2] += W17[:, idx1]
                # Remove term idx1
                if idx1 < idx2:
                    U17 = np.delete(U17, idx1, axis=1)
                    V17 = np.delete(V17, idx1, axis=1)
                    W17 = np.delete(W17, idx1, axis=1)
                else:
                    U17 = np.delete(U17, idx2, axis=1)
                    V17 = np.delete(V17, idx2, axis=1)
                    W17 = np.delete(W17, idx2, axis=1)
                
                if U17.shape[1] == 17:
                    U_r, V_r, W_r, success = _als_rank_search(17, U17, V17, W17, max_iter=100)
                    if success:
                        best_rank = 17
                        best_decomp = (U_r, V_r, W_r)
                        break
    
    U, V, W = best_decomp
    
    # Round to nearest integer if close
    def _round_if_close(arr, tol=1e-6):
        rounded = np.round(arr)
        if np.allclose(arr, rounded, atol=tol, rtol=0):
            return rounded
        return arr
    
    U = _round_if_close(U)
    V = _round_if_close(V)
    W = _round_if_close(W)
    
    return {"U": U.astype(int).tolist(), "V": V.astype(int).tolist(), "W": W.astype(int).tolist()}

# EVOLVE-BLOCK-END
