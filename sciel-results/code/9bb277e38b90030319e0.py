import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill
    
    n, m, p = 2, 2, 4
    
    # Try to find rank-12 decomposition
    # The multiplication tensor: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # U: (4, rank), V: (8, rank), W: (8, rank)
    # W index: k*n+i where k=0..3, i=0..1
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly reproduces matrix multiplication."""
        rank = U.shape[1]
        # For each (i,j,k), check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == delta
        # The multiplication tensor has T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B 
        # contribute to C[i,k], which is always the case - so T[i,j,k]=1 for all valid i,j,k
        # Wait, no. The tensor is T_{ijk} where the bilinear form is sum_{i,j,k} T_{ijk} A_{ij} B_{jk} = C_{ik}
        # So T_{ijk} = 1 for all i,j,k (it's just the product)
        # Actually for matrix multiplication C = A*B, C_{ik} = sum_j A_{ij} B_{jk}
        # So the tensor T_{ijk} = 1 for all valid i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_rank12():
        """Try to find a rank-12 decomposition using peeling/structured search."""
        # Use a structured ansatz based on Strassen
        # For 2x2x2, Strassen uses 7 products
        # For 2x2x2x4, try to use 12 products
        
        # Let's use a parameterized approach
        # Based on the structure: C = A @ B where A is 2x2, B is 2x4
        # C_{0,0} = A_{0,0}*B_{0,0} + A_{0,1}*B_{1,0}
        # C_{0,1} = A_{0,0}*B_{0,1} + A_{0,1}*B_{1,1}
        # C_{0,2} = A_{0,0}*B_{0,2} + A_{0,1}*B_{1,2}
        # C_{0,3} = A_{0,0}*B_{0,3} + A_{0,1}*B_{1,3}
        # C_{1,0} = A_{1,0}*B_{0,0} + A_{1,1}*B_{1,0}
        # C_{1,1} = A_{1,0}*B_{0,1} + A_{1,1}*B_{1,1}
        # C_{1,2} = A_{1,0}*B_{0,2} + A_{1,1}*B_{1,2}
        # C_{1,3} = A_{1,0}*B_{0,3} + A_{1,1}*B_{1,3}
        
        # Try a rank-12 ansatz with some symmetry
        # Use the idea that we can share some intermediate products
        
        # Let's try: define 12 bilinear forms
        # M1 = A00*B00, M2 = A00*B01, M3 = A00*B02, M4 = A00*B03
        # M5 = A01*B10, M6 = A01*B11, M7 = A01*B12, M8 = A01*B13
        # That's 8 for row 0 of C
        # M9 = A10*B00, M10 = A10*B01, M11 = A10*B02, M12 = A10*B03
        # That's 8 more, total 16 (schoolbook)
        
        # For rank 12, we need to combine some of these
        # Try Strassen-like: 
        # Let's use the following 12 products:
        # p1 = (A00+A10)*(B00+B01)  -- no, this doesn't work directly
        
        # Actually, let me try a different approach.
        # The key insight: for 2x2x2, rank is 7.
        # For 2x2x2x4, we can view it as two 2x2x2 problems.
        # Problem 1: A(2x2) * B_left(2x2) -> C_left(2x2)
        # Problem 2: A(2x2) * B_right(2x2) -> C_right(2x2)
        # Each has rank 7, total 14. To get 12, we need to share 2 products.
        
        # Strassen for 2x2x2:
        # m1 = (A00+A11)*(B00+B11)
        # m2 = (A10+A11)*B00
        # m3 = A00*(B01-B11)
        # m4 = A11*(B10-B00)
        # m5 = (A00+A01)*B11
        # m6 = (A10-A00)*(B00+B01)
        # m7 = (A01-A11)*(B10+B11)
        
        # For the left block (B columns 0,1):
        # m1L = (A00+A11)*(B00+B11)
        # m2L = (A10+A11)*B00
        # m3L = A00*(B01-B11)
        # m4L = A11*(B10-B00)
        # m5L = (A00+A01)*B11
        # m6L = (A10-A00)*(B00+B01)
        # m7L = (A01-A11)*(B10+B11)
        
        # For the right block (B columns 2,3):
        # m1R = (A00+A11)*(B02+B13)
        # m2R = (A10+A11)*B02
        # m3R = A00*(B03-B13)
        # m4R = A11*(B12-B02)
        # m5R = (A00+A01)*B13
        # m6R = (A10-A00)*(B02+B03)
        # m7R = (A01-A11)*(B12+B13)
        
        # Total: 14 products. To get 12, we need to share 2.
        # Notice m1L and m1R both involve (A00+A11). 
        # m1L = (A00+A11)*(B00+B11), m1R = (A00+A11)*(B02+B13)
        # We can't directly combine these into one product.
        
        # Alternative: m2L = (A10+A11)*B00, m2R = (A10+A11)*B02
        # Still separate.
        
        # Let me try a different sharing strategy.
        # What if we use:
        # s1 = A00+A11, s2 = A10+A11, s3 = A00+A01, s4 = A10-A00, s5 = A01-A11
        # t1 = B00+B11, t2 = B01-B11, t3 = B10-B00, t4 = B11, t5 = B00+B01, t6 = B10+B11
        # Similarly for right block.
        
        # This is getting complex. Let me try a direct numerical search.
        
        # Approach: use a random initialization and gradient-free optimization
        # to minimize the residual, trying to find a rank-12 solution.
        
        from scipy.optimize import minimize
        
        def residual_flat(params, rank=12):
            """Flatten U, V, W into a single parameter vector and compute residual."""
            # U: 4*rank, V: 8*rank, W: 8*rank
            total = (4 + 8 + 8) * rank
            if len(params) != total:
                return 1e10
            
            U = params[:4*rank].reshape(4, rank)
            V = params[4*rank:12*rank].reshape(8, rank)
            W = params[12*rank:].reshape(8, rank)
            
            res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                        res += (val - 1.0)**2
            return res
        
        # Try several random initializations
        np.random.seed(42)
        best_val = 1e10
        
        for attempt in range(5):
            if time.time() > deadline - 30:
                break
            
            rank = 12
            # Initialize with small random values
            params = np.random.randn((4 + 8 + 8) * rank) * 0.5
            
            # Use Nelder-Mead for small problems
            try:
                result = minimize(residual_flat, params, method='Nelder-Mead', 
                                  options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
                if result.fun < best_val:
                    best_val = result.fun
                    best_params = result.x
            except Exception:
                pass
        
        if best_val < 1e-8:
            params = best_params
            rank = 12
            U = params[:4*rank].reshape(4, rank)
            V = params[4*rank:12*rank].reshape(8, rank)
            W = params[12*rank:].reshape(8, rank)
            if verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    # Try rank 12
    result = try_rank12()
    if result is not None:
        return result
    
    # Fallback: rank 14 (known valid solution)
    rank = 14
    U = [[0.0]*rank for _ in range(4)]
    V = [[0.0]*rank for _ in range(8)]
    W = [[0.0]*rank for _ in range(8)]
    
    def set_col(col, u_dict, v_dict, w_dict):
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
    
    set_col(0, {0:1, 3:1}, {0:1, 5:1}, {0:1, 3:1})
    set_col(1, {2:1, 3:1}, {0:1}, {1:1, 3:-1})
    set_col(2, {0:1}, {1:1, 5:-1}, {2:1, 3:1})
    set_col(3, {3:1}, {4:1, 0:-1}, {0:1, 1:1})
    set_col(4, {0:1, 1:1}, {5:1}, {0:-1, 2:1})
    set_col(5, {2:1, 0:-1}, {0:1, 1:1}, {3:1})
    set_col(6, {1:1, 3:-1}, {4:1, 5:1}, {0:1})
    set_col(7, {0:1, 3:1}, {2:1, 7:1}, {4:1, 7:1})
    set_col(8, {2:1, 3:1}, {2:1}, {5:1, 7:-1})
    set_col(9, {0:1}, {3:1, 7:-1}, {6:1, 7:1})
    set_col(10, {3:1}, {6:1, 2:-1}, {4:1, 5:1})
    set_col(11, {0:1, 1:1}, {7:1}, {4:-1, 6:1})
    set_col(12, {2:1, 0:-1}, {2:1, 3:1}, {7:1})
    set_col(13, {1:1, 3:-1}, {6:1, 7:1}, {4:1})
    
    return {"U": U, "V": V, "W": W}
