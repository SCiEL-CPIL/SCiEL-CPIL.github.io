<<<<<<< SEARCH
"""Matrix multiplication 3x5x6 tensor rank minimization via structured search."""
import time
import random
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]
_START_TIME = time.time()
_DEADLINE = 150.0
_SAFE_MARGIN = 5.0

def _time_remaining():
    return _DEADLINE - _SAFE_MARGIN - (time.time() - _START_TIME)

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix
    by a 5x6 matrix. Output factor W follows W[k*n+i, r].
    Returns {'U', 'V', 'W'} where:
      U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 90
    
    # Known lower bounds
    # Strassen: rank >= 2^{ceil(log2 n)} * 2^{ceil(log2 m)} * 2^{ceil(log2 p)} roughly
    # But for mixed sizes, we need better bounds
    # Known results: R(3,5,6) is bounded above by various algorithms
    
    # Strategy: Start from known good decompositions or schoolbook,
    # then attempt to reduce rank via structured random search with exact arithmetic.
    
    # The tensor is defined by: T[i,j,k] = 1 if C[i,k] = sum_j A[i,j]*B[j,k]
    # Factorization: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # We'll work with the matrix multiplication tensor entries.
    # T[(i,j,k)] = 1 for all valid i,j,k (it's the structure tensor)
    # Actually, the tensor maps A,B -> C where C[i,k] = sum_j A[i,j]B[j,k]
    # So the structure tensor has T[i,j,k] = 1 for all i,j,k in their ranges.
    # Wait, no. The tensor T is defined such that:
    # C[i,k] = sum_{j,r} A[i,j] * U[j, r] ... no.
    
    # Let me re-derive:
    # We want: C[i,k] = sum_j A[i,j] * B[j,k]
    # Factored form: C[i,k] = sum_r (sum_j A[i,j] * U[j * ? , r]) ... 
    
    # The contract says:
    # U has shape (n*m, rank) - indexed by [i*m + j, r]
    # V has shape (m*p, rank) - indexed by [j*p + k, r]  
    # W has shape (p*n, rank) - indexed by [k*n + i, r]
    
    # And the decomposition must satisfy:
    # For all i,j,k: delta[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where delta[i,j,k] = 1 (since the multiplication tensor has all ones as structure constants
    # in the appropriate basis)
    
    # Actually, the tensor for matrix multiplication A (n x m) * B (m x p) -> C (n x p)
    # has structure tensor entries: T[i, (j,k)] = A[i,j]*B[j,k] summed over j
    # In tensor form: T[i,j,k] = 1 (the structure constants are all 1 in the standard basis)
    
    # The rank-1 decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For the schoolbook: U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for each (i,j,k)
    
    # We need to find a lower rank decomposition.
    
    # Approach: Use the known fact that for small matrices, we can sometimes do better than schoolbook.
    # Known results:
    # R(2,2,2) = 7 (Strassen)
    # R(3,3,3) <= 23 (Winograd)
    # R(2,3,3) = 13? Not sure.
    # R(3,4,4) <= ?
    
    # Let's try a randomized search approach with exact rational arithmetic.
    # Start with schoolbook and try to find linear dependencies that allow rank reduction.
    
    # Build the structure tensor as a 3D array for verification
    def make_tensor(n, m, p):
        T = np.ones((n, m, p), dtype=np.int64)
        return T
    
    T_np = make_tensor(n, m, p)
    
    # Verification function using exact arithmetic
    def verify_exact(U_mat, V_mat, W_mat, n, m, p):
        """U_mat: (n*m, rank), V_mat: (m*p, rank), W_mat: (p*n, rank) as lists of lists of Fraction"""
        rank = len(U_mat[0]) if U_mat else 0
        # For each (i,j,k), check sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    ur = i * m + j
                    vr = j * p + k
                    wr = k * n + i
                    for r in range(rank):
                        total += U_mat[ur][r] * V_mat[vr][r] * W_mat[wr][r]
                    if total != 1:
                        return False
        return True
    
    # Convert to/from fraction helpers
    def to_frac_matrix(mat_list):
        """Convert list of lists of numbers to list of lists of Fraction"""
        return [[Fraction(x) for x in row] for row in mat_list]
    
    def from_frac_matrix(fmat):
        """Convert list of lists of Fraction to list of lists of int or Fraction"""
        result = []
        for row in fmat:
            out_row = []
            for x in row:
                if x.denominator == 1:
                    out_row.append(x.numerator)
                else:
                    out_row.append(x)
            result.append(out_row)
        return result
    
    # Initialize with schoolbook
    def schoolbook_decomposition(n, m, p):
        rank = n * m * p
        U = [[Fraction(0)] * rank for _ in range(n * m)]
        V = [[Fraction(0)] * rank for _ in range(m * p)]
        W = [[Fraction(0)] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = Fraction(1)
                    V[j * p + k][col] = Fraction(1)
                    W[k * n + i][col] = Fraction(1)
                    col += 1
        return U, V, W
    
    U_frac, V_frac, W_frac = schoolbook_decomposition(n, m, p)
    
    # Try to reduce rank by finding a smaller decomposition.
    # Strategy: Randomized local search.
    # 1. Try to express one rank-1 term as a combination of others (unlikely to help directly)
    # 2. Use the tensor network approach: treat it as a matrix equation.
    
    # Better approach: Use the fact that the tensor can be reshaped.
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    # The constraint is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    
    # This is a system of n*m*p equations in rank*(n*m + m*p + p*n) variables.
    # We want the minimum rank.
    
    # Let's try a greedy/randomized approach:
    # Start with rank = some guess, and try to solve the system.
    
    # Known upper bounds for R(3,5,6):
    # Strassen-type: Not directly applicable for non-power-of-2
    # Padua et al. and others have computed bounds.
    # For 3x5x6, the schoolbook is 90.
    # A simple improvement: split 5 = 2+3, use Strassen for 2x2x2 parts
    # But let's try a more systematic approach.
    
    # Approach: Randomized search over small rank with exact arithmetic.
    # We'll parameterize with rational numbers and use a solver-like approach.
    
    # Actually, let's try a different strategy:
    # Use the known result that R(3,4,4) <= 23? No, that's R(3,3,3) <= 23.
    # For our case, let's just try to find ANY decomposition with rank < 90.
    
    # Quick win: Can we factor the schoolbook?
    # The schoolbook has rank 90. Let's see if there's a simple structure.
    
    # Alternative: Use numpy to compute a numerical rank approximation, then round.
    # This is a heuristic approach.
    
    # Let's try a randomized approach with a time limit.
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_frac, V_frac, W_frac
    
    # Try various rank values
    # Lower bound: At minimum, we need rank >= max(n*m, m*p, p*n) / something
    # Actually, a simple lower bound: rank >= ceil((n*m*p) / min(n*m, m*p, p*n)) 
    # That's not right either.
    
    # For matrix multiplication tensor, a known lower bound is:
    # R(n,m,p) >= n*m*p / max(n,m,p) ? Not sure.
    
    # Let's just try to find a decomposition with rank as small as possible.
    
    # Use a random initialization and Gauss-Newton or similar with exact arithmetic.
    # This is complex. Let's try a simpler approach:
    
    # Try rank values from a lower bound up to schoolbook, and for each,
    # try to solve the nonlinear system using random restarts.
    
    # For small systems, we can enumerate or use random sampling.
    
    # Key insight: For the structure tensor T[i,j,k]=1, the decomposition
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # Let's fix the problem structure. Note that W[k*n+i, r] depends on k and i.
    # Let's define W' [i,k,r] = W[k*n+i, r], so the constraint is:
    # sum_r U[i,j,r] * V[j,k,r] * W'[i,k,r] = 1 for all i,j,k
    
    # This is a 3-tensor CP decomposition with specific index coupling.
    
    # Randomized search: For a given rank r, initialize U, V, W' randomly with small rationals,
    # then use a local search to minimize the error.
    
    # But exact arithmetic with random rationals is tricky. Let's use integers.
    
    # Alternative approach: Use the known Strassen algorithm adapted.
    # For 3x5x6: split into blocks.
    # 3 = 2+1, 5 = 2+3, 6 = 3+3
    # This gets complicated.
    
    # Let's just try a direct numerical approach and round.
    
    def try_rank(rank, max_iters=100, seed=42):
        """Try to find a decomposition of given rank using random restarts."""
        if rank < 1:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with small random integers
        U_init = rng.randint(-1, 2, (n*m, rank))
        V_init = rng.randint(-1, 2, (m*p, rank))
        W_init = rng.randint(-1, 2, (p*n, rank))
        
        # Check if any random init works (extremely unlikely)
        for restart in range(max_iters):
            U_r = U_init.copy() if restart == 0 else rng.randint(-1, 2, (n*m, rank))
            V_r = V_init.copy() if restart == 0 else rng.randint(-1, 2, (m*p, rank))
            W_r = W_init.copy() if restart == 0 else rng.randint(-1, 2, (p*n, rank))
            
            # Quick numerical check
            T_check = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0
                        for r in range(rank):
                            total += U_r[i*m+j, r] * V_r[j*p+k, r] * W_r[k*n+i, r]
                        T_check[i,j,k] = total
            
            if np.allclose(T_check, 1.0, atol=1e-10):
                # Convert to fractions
                U_f = to_frac_matrix(U_r.tolist())
                V_f = to_frac_matrix(V_r.tolist())
                W_f = to_frac_matrix(W_r.tolist())
                if verify_exact(U_f, V_f, W_f, n, m, p):
                    return U_f, V_f, W_f
        
        return None
    
    # Try decreasing ranks
    # Start from a reasonable upper bound
    # For 3x5x6, known bounds might be around 60-80
    # Let's try a range
    
    ranks_to_try = []
    # Add schoolbook
    ranks_to_try.append(schoolbook_rank)
    # Try some lower ranks
    for r in range(80, 89):
        ranks_to_try.insert(0, r)
    
    # Sort and try
    for rank in sorted(set(ranks_to_try)):
        if _time_remaining() < 10:
            break
        
        result = try_rank(rank, max_iters=5, seed=rank)
        if result is not None:
            U_frac, V_frac, W_frac = result
            best_rank = rank
            break
    
    # If we haven't improved, try a more sophisticated approach
    if best_rank == schoolbook_rank and _time_remaining() > 30:
        # Try a structured approach: use known small-matrix results
        # For 2x2x2, Strassen gives rank 7.
        # Can we build a 3x5x6 from Strassen blocks?
        
        # Let's try to exploit the fact that 3 = 2+1, etc.
        # This is getting complex. Let's just return the best we have.
        
        # Actually, let me try a different numerical approach:
        # Set up the system of equations and use least squares.
        
        def solve_numerical(rank):
            """Use numerical optimization to find a decomposition."""
            # Total parameters: rank * (n*m + m*p + p*n)
            np_params = rank * (n*m + m*p + p*n)
            
            # Objective: minimize sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
            # This is nonlinear. Use random init and gradient-free optimization.
            
            best_error = np.inf
            best_params = None
            
            for attempt in range(20):
                # Random init
                params = np.random.randn(np_params) * 0.1
                
                # Try to improve with a few iterations of a simple local search
                # This is very slow. Skip for now.
                
                # Just check the random init
                U_mat = params[:rank*n*m].reshape((n*m, rank))
                V_mat = params[rank*n*m:rank*(n*m+m*p)].reshape((m*p, rank))
                W_mat = params[rank*(n*m+m*p):].reshape((p*n, rank))
                
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = 0
                            for r in range(rank):
                                total += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                            error += (total - 1)**2
                
                if error < best_error:
                    best_error = error
                    best_params = params.copy()
            
            # Convert best_params to fractions by rounding to nearest rational with small denominator
            if best_params is not None:
                U_mat = best_params[:rank*n*m].reshape((n*m, rank))
                V_mat = best_params[rank*n*m:rank*(n*m+m*p)].reshape((m*p, rank))
                W_mat = best_params[rank*(n*m+m*p):].reshape((p*n, rank))
                
                # Round to nearest multiple of 1/4 or 1/8
                def round_rational(x):
                    # Try denominators 1, 2, 3, 4
                    for den in [1, 2, 3, 4]:
                        num = round(x * den)
                        if abs(x * den - num) < 1e-6:
                            return Fraction(num, den)
                    return Fraction(x).limit_denominator(8)
                
                U_f = [[round_rational(x) for x in row] for row in U_mat]
                V_f = [[round_rational(x) for x in row] for row in V_mat]
                W_f = [[round_rational(x) for x in row] for row in W_mat]
                
                if verify_exact(U_f, V_f, W_f, n, m, p):
                    return U_f, V_f, W_f
            
            return None
        
        # Try some ranks
        for rank in range(60, 70):
            if _time_remaining() < 5:
                break
            result = solve_numerical(rank)
            if result is not None:
                U_frac, V_frac, W_frac = result
                best_rank = rank
                break
    
    # Return the best result found
    result_U = from_frac_matrix(U_frac)
    result_V = from_frac_matrix(V_frac)
    result_W = from_frac_matrix(W_frac)
    
    return {"U": result_U, "V": result_V, "W": result_W}
=======
"""Matrix multiplication 3x5x6 tensor rank minimization via structured search."""
import time
import random
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]
_START_TIME = time.time()
_DEADLINE = 150.0
_SAFE_MARGIN = 5.0

def _time_remaining():
    return _DEADLINE - _SAFE_MARGIN - (time.time() - _START_TIME)

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix
    by a 5x6 matrix. Output factor W follows W[k*n+i, r].
    Returns {'U', 'V', 'W'} where:
      U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 90
    
    # Verification function using exact arithmetic
    def verify_exact(U_mat, V_mat, W_mat, n, m, p):
        """U_mat: (n*m, rank), V_mat: (m*p, rank), W_mat: (p*n, rank) as lists of lists of Fraction"""
        rank = len(U_mat[0]) if U_mat else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    ur = i * m + j
                    vr = j * p + k
                    wr = k * n + i
                    for r in range(rank):
                        total += U_mat[ur][r] * V_mat[vr][r] * W_mat[wr][r]
                    if total != 1:
                        return False
        return True
    
    def from_frac_matrix(fmat):
        """Convert list of lists of Fraction to list of lists of int or Fraction"""
        result = []
        for row in fmat:
            out_row = []
            for x in row:
                if x.denominator == 1:
                    out_row.append(x.numerator)
                else:
                    out_row.append(x)
            result.append(out_row)
        return result
    
    # Strategy: Block-wise Kronecker product decomposition
    # Treat 3x5 matrix as 1x3 block row of 1x5 vectors
    # Treat 5x6 matrix as 5x1 block column of 1x6 vectors
    # The base tensor is 1x5 x 5x6 -> 1x6, which has rank 5 (schoolbook for vector outer product)
    # The full tensor is the Kronecker product with the 3x1 scalar identity
    
    # For the base 1x5 * 5x6 -> 1x6 tensor:
    # T_base[j, k] = 1 for all j in range(5), k in range(6)
    # Schoolbook decomposition has rank 5:
    # U_base[j, r] = 1 if j == r else 0 (for r in 0..4)
    # V_base[j, k, r] = 1 if j == r else 0
    # W_base[k, r] = 1 (all ones)
    
    # Wait, let me re-derive. For 1x5 * 5x6 -> 1x6:
    # C[0, k] = sum_j A[0, j] * B[j, k]
    # In tensor form: T[i, j, k] where i is always 0
    # T[0, j, k] = 1 for all j, k
    # Rank-1 term: U[0*m+j, r] * V[j*p+k, r] * W[k*n+0, r]
    # With n=1, m=5, p=6:
    # U[j, r] * V[j*6+k, r] * W[k, r] = 1 for all j, k
    
    # Schoolbook: rank = 5*6 = 30? No, that's for the full 1x5x6.
    # Actually for 1x5 * 5x6, the schoolbook has rank 5*6 = 30? 
    # No: C[0,k] = sum_j A[0,j]*B[j,k], so each (j,k) pair is one multiplication.
    # That's 5*6 = 30 terms. But we can do better!
    
    # Actually, this is just an outer product. C = A * B where A is 1x5, B is 5x6.
    # C[0,k] = sum_j A[0,j] B[j,k]
    # This is a rank-1 operation in the sense that it's a single vector dot product chain.
    # The tensor T[j,k] = 1 (for the structure) has rank... 
    
    # Let me think differently. The multiplication tensor for (1,5,6) is:
    # T[j,k] = 1 (structure constant)
    # We need: sum_r U[j,r] * V[j*6+k,r] * W[k,r] = 1 for all j,k
    
    # Schoolbook: for each (j,k), set U[j,r]=1, V[j*6+k,r]=1, W[k,r]=1 with unique r
    # That gives rank 30.
    
    # Can we do better? Note that W[k,r] appears for all j.
    # If we set W[k,r] = 1 for all k,r, then we need sum_r U[j,r]*V[j*6+k,r] = 1
    # This is like a matrix factorization: let M[j,k] = sum_r U[j,r]*V[j*6+k,r]
    # We need M[j,k] = 1 for all j,k.
    # If U is 5xR and V reshaped as 5x6xR, then M[j,k] = sum_r U[j,r]*V_reshaped[j,k,r]
    # This is not standard matrix multiplication.
    
    # Alternative: Set U[j,r] = delta(j,r), V[j*6+k,r] = delta(j,r), W[k,r] = 1
    # Then sum_r delta(j,r)*delta(j,r)*1 = delta(j,j) = 1 if j in range. That works for rank 5!
    # Check: for fixed j,k, sum_r U[j,r]*V[j*6+k,r]*W[k,r] = sum_r delta(j,r)*delta(j,r)*1 = 1 (since r=j)
    # Yes! So the 1x5x6 tensor has rank 5.
    
    # Now for the full 3x5x6 tensor:
    # T[i,j,k] = 1 for all i in 0..2, j in 0..4, k in 0..5
    # We need: sum_r U[i*5+j, r] * V[j*6+k, r] * W[k*3+i, r] = 1
    
    # Kronecker product approach:
    # Let the base 1x5x6 decomposition have rank 5 with factors U0, V0, W0
    # U0[j, r] = delta(j, r) for r in 0..4
    # V0[j*6+k, r] = delta(j, r) for r in 0..4
    # W0[k, r] = 1 for all k, r
    
    # For the 3x5x6 case, we can use the Kronecker product with the 3x1x1 "identity"
    # The 3x1x1 tensor is just the identity on the first index, rank 1.
    # U_id[i, r] = delta(i, r) for r in 0..2? No, that doesn't work directly.
    
    # Actually, let's think of it as: the 3x5x6 tensor is the tensor product of
    # the 3x1x1 identity tensor (rank 1) and the 1x5x6 tensor (rank 5).
    # The tensor product of two rank-r and rank-s tensors has rank r*s.
    # So the full tensor has rank 1*5 = 5? That seems too good.
    
    # Wait, the 3x1x1 tensor: T_id[i, 0, 0] = 1 for i in 0..2.
    # Rank-1 decomposition: U_id[i, r] * V_id[0, r] * W_id[0, r] = 1 for all i.
    # Set U_id[i, 0] = 1, V_id[0, 0] = 1, W_id[0, 0] = 1. Rank 1.
    
    # The tensor product of T_id (3x1x1) and T_base (1x5x6) gives T_full (3x5x6).
    # T_full[i, j, k] = T_id[i, 0, 0] * T_base[0, j, k] = 1 * 1 = 1. Correct!
    
    # The rank of the tensor product is the product of the ranks: 1 * 5 = 5.
    # So we can decompose 3x5x6 with rank 5!
    
    # Let's construct it explicitly.
    # T_id has rank 1: U_id[i, 0] = 1, V_id[0, 0] = 1, W_id[0, 0] = 1
    # T_base has rank 5: U_base[j, r] = delta(j, r), V_base[j*6+k, r] = delta(j, r), W_base[k, r] = 1
    
    # The tensor product decomposition:
    # For each pair (r_id, r_base), we get a rank-1 term.
    # Since r_id is always 0, we just have r_base terms.
    
    # U_full[i*5+j, r] = U_id[i, 0] * U_base[j, r] = 1 * delta(j, r) = delta(j, r)
    # V_full[j*6+k, r] = V_id[0, 0] * V_base[j*6+k, r] = 1 * delta(j, r) = delta(j, r)
    # W_full[k*3+i, r] = W_id[0, 0] * W_base[k, r] = 1 * 1 = 1
    
    # Let's verify: sum_r U_full[i*5+j, r] * V_full[j*6+k, r] * W_full[k*3+i, r]
    # = sum_r delta(j, r) * delta(j, r) * 1
    # = delta(j, j) = 1 (since j is in 0..4, and r=j is in the sum)
    # Yes! This works with rank 5.
    
    rank = 5
    U_frac = [[Fraction(0)] * rank for _ in range(n * m)]
    V_frac = [[Fraction(0)] * rank for _ in range(m * p)]
    W_frac = [[Fraction(0)] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            U_frac[i * m + j][j] = Fraction(1)
    
    for j in range(m):
        for k in range(p):
            V_frac[j * p + k][j] = Fraction(1)
    
    for k in range(p):
        for i in range(n):
            W_frac[k * n + i][0] = Fraction(0)  # Will set below
    
    # Actually, W_full[k*3+i, r] = 1 for all k, i, r? No, that's not right.
    # Let me re-check the tensor product formula.
    
    # For tensor product of T1 (dims a1,b1,c1) and T2 (dims a2,b2,c2):
    # T1[i1,j1,k1] * T2[i2,j2,k2] -> T[i1*a2+i2, j1*b2+j2, k1*c2+k2]
    # Rank-1 terms: if T1 = sum_r u1[r] v1[r] w1[r] and T2 = sum_s u2[s] v2[s] w2[s]
    # Then T = sum_{r,s} (u1[r]⊗u2[s]) (v1[r]⊗v2[s]) (w1[r]⊗w2[s])
    
    # For T_id (3x1x1): u1[i, 0] = 1, v1[0, 0] = 1, w1[0, 0] = 1
    # For T_base (1x5x6): u2[j, r] = delta(j,r), v2[j*6+k, r] = delta(j,r), w2[k, r] = 1
    
    # T_full dims: (3*1, 5*6, 6*1) = (3, 30, 6)? That's not right.
    
    # I think I'm confusing the tensor product with the Khatri-Rao or something else.
    # Let me just directly construct the rank-5 decomposition I derived above.
    
    # U[i*5+j, r] = delta(j, r) for r in 0..4
    # V[j*6+k, r] = delta(j, r) for r in 0..4
    # W[k*3+i, r] = ? 
    
    # From the verification: sum_r U[i*5+j, r] * V[j*6+k, r] * W[k*3+i, r] = 1
    # = sum_r delta(j, r) * delta(j, r) * W[k*3+i, r]
    # = delta(j, j) * W[k*3+i, j]
    # = W[k*3+i, j]
    
    # So we need W[k*3+i, j] = 1 for all k, i, j.
    # That means W is all ones!
    
    for k in range(p):
        for i in range(n):
            for r in range(rank):
                W_frac[k * n + i][r] = Fraction(1)
    
    # Verify
    if not verify_exact(U_frac, V_frac, W_frac, n, m, p):
        # Fallback to schoolbook
        rank = schoolbook_rank
        U_frac = [[Fraction(0)] * rank for _ in range(n * m)]
        V_frac = [[Fraction(0)] * rank for _ in range(m * p)]
        W_frac = [[Fraction(0)] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_frac[i * m + j][col] = Fraction(1)
                    V_frac[j * p + k][col] = Fraction(1)
                    W_frac[k * n + i][col] = Fraction(1)
                    col += 1
    
    result_U = from_frac_matrix(U_frac)
    result_V = from_frac_matrix(V_frac)
    result_W = from_frac_matrix(W_frac)
    
    return {"U": result_U, "V": result_V, "W": result_W}
>>>>>>> REPLACE
