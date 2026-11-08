import time
from fractions import Fraction
import random

def construct():
    t0 = time.time()
    deadline = t0 + 145.0

    n, m, p = 2, 3, 8

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Strassen 2x2x2
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    # Build initial rank-44 solution
    rank0 = 44
    U = [[0] * rank0 for _ in range(n * m)]
    V = [[0] * rank0 for _ in range(m * p)]
    W = [[0] * rank0 for _ in range(p * n)]

    col = 0
    # Part 1: A1 (2x2) * B1 (2x8), split B1 into 4 blocks of 2
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U[li * m + la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V[la * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W[(k0 + lk) * n + li][col] = w_val
            col += 1

    # Part 2: A2 (2x1) * B2 (1x8)
    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Try to reduce rank by finding linear dependencies
    # We attempt to find a smaller rank decomposition by trying to express
    # some columns as linear combinations of others
    def _try_reduce_rank(U, V, W, target_rank):
        """
        Try to find a rank-`target_rank` decomposition by checking if we can
        eliminate (rank0 - target_rank) columns.
        """
        rank0 = len(U[0])
        
        # For each column c, check if it's "redundant" - i.e., its contribution
        # can be replicated by a linear combination of other columns.
        
        # We'll try greedy elimination: for each column to remove, check if
        # the tensor equation still holds.
        
        # Build the tensor equation residual
        # T[i,a,k] = sum_c U[i,a,c] * V[a,k,c] * W[k,i,c]
        
        # We want to find coefficients alpha_c such that for each column c_removed,
        # the "column vector" (U[:,c], V[:,c], W[:,c]) is in the span of others.
        
        # Simpler approach: try removing one column at a time and see if we can
        # fix the others.
        
        # Actually, let's try a different approach: check if any column has
        # a pattern that's linearly dependent on others.
        
        # For a column c, define the "signature" as the triple of vectors
        # (U[:,c], V[:,c], W[:,c]). Two columns c1, c2 are linearly dependent
        # if U[:,c1] = alpha*U[:,c2], V[:,c1] = alpha*V[:,c2], W[:,c1] = alpha*W[:,c2]
        
        # Check for exact linear dependencies
        def _col_vectors(U, V, W, c):
            return (U[:, c], V[:, c], W[:, c])
        
        # Try to find pairs of columns that are scalar multiples
        for c_remove in range(rank0):
            if not _elapsed_ok():
                break
            
            # Check if column c_remove is a linear combination of other columns
            # This is complex, so try simple cases: is it zero?
            u_col = [U[ia][c_remove] for ia in range(len(U))]
            v_col = [V[ak][c_remove] for ak in range(len(V))]
            w_col = [W[ki][c_remove] for ki in range(len(W))]
            
            # Check if all are zero
            if all(x == 0 for x in u_col) and all(x == 0 for x in v_col) and all(x == 0 for x in w_col):
                # Remove this column
                new_U = [row[:c_remove] + row[c_remove+1:] for row in U]
                new_V = [row[:c_remove] + row[c_remove+1:] for row in V]
                new_W = [row[:c_remove] + row[c_remove+1:] for row in W]
                return {"U": new_U, "V": new_V, "W": new_W}
            
            # Check if it's a scalar multiple of another column
            for c_other in range(rank0):
                if c_other == c_remove:
                    continue
                
                u_other = [U[ia][c_other] for ia in range(len(U))]
                v_other = [V[ak][c_other] for ak in range(len(V))]
                w_other = [W[ki][c_other] for ki in range(len(W))]
                
                # Find scalar alpha such that u_col = alpha * u_other
                alpha = None
                for ia in range(len(U)):
                    if u_other[ia] != 0:
                        alpha = Fraction(u_col[ia], u_other[ia])
                        break
                
                if alpha is None:
                    continue
                
                # Check if u_col = alpha * u_other
                if any(u_col[ia] != alpha * u_other[ia] for ia in range(len(U))):
                    continue
                # Check if v_col = alpha * v_other
                if any(v_col[ak] != alpha * v_other[ak] for ak in range(len(V))):
                    continue
                # Check if w_col = alpha * w_other
                if any(w_col[ki] != alpha * w_other[ki] for ki in range(len(W))):
                    continue
                
                # Column c_remove is a scalar multiple of c_other, so we can remove it
                # (since the tensor is multilinear, if column c = alpha * column d,
                # then the contribution of c is alpha times the contribution of d,
                # but we need the sum to remain the same... actually this means
                # the columns are not independent in the tensor decomposition)
                
                # Wait, this isn't quite right. In a tensor decomposition, each column
                # contributes independently. If column c = alpha * column d (as a triple),
                # then the contribution from c is alpha * (contribution from d), which
                # means we can't simply remove it unless we adjust d.
                
                # Actually, the tensor is T = sum_c U[:,c] V[:,c] W[:,c].
                # If (U[:,c], V[:,c], W[:,c]) = alpha * (U[:,d], V[:,d], W[:,d]),
                # then the term for c is alpha * (term for d).
                # So T = ... + alpha * term_d + ...
                # We can't remove c without changing T unless we also modify d.
                
                # So this simple check doesn't work directly. Let me think differently.
                
                pass

        return None

    # Better approach: use the fact that we have a rank-44 decomposition and try
    # to find a rank-43 or lower decomposition by solving linear equations.
    
    # Let's try a randomized approach: randomly set some entries to 0 and see
    # if we can still satisfy the tensor equations.
    
    def _random_reduction_attempt(U, V, W, num_trials=10000):
        rank0 = len(U[0])
        best = None
        
        rng = random.Random(42)
        
        for trial in range(num_trials):
            if not _elapsed_ok():
                break
            
            # Try to remove one column
            c_remove = rng.randint(0, rank0 - 1)
            
            # Check if we can adjust the remaining columns to compensate
            # This is hard, so instead let's just check if the column is "small"
            
            # Alternative: try a different decomposition structure
            # For 2x3x8, maybe we can do better than 44.
            
            # Let's try: 2x2x2 uses 7, 2x2x2 uses 7, 2x2x2 uses 7, 2x2x2 uses 7 = 28
            # Plus 2x1x8 = 16
            # Total = 44
            # Can we do 2x3x8 more cleverly?
            
            # What if we use a 3x3x3 block? No, the matrix is 2x3, not 3x3.
            
            # What if we group the 3 rows of A differently?
            # A is 2x3, B is 3x8.
            # We could split A into 2x1 and 2x2 (as we did), or 1x3 and 1x3.
            
            # Let's try: split into two 1x3 blocks (each 1 row of A times all 3 columns)
            # No, A has 2 rows, so we can't split into 1x3 blocks cleanly.
            
            pass
        
        return best

    # Let's try a completely different decomposition:
    # Use rank-7 for 2x2x2 blocks, but group the 8 columns of B into 4 pairs.
    # We already did this, giving 4*7 + 2*8 = 28 + 16 = 44.
    
    # What if we use a different grouping? For example, 2x2x4 blocks?
    # 2x2x4 rank is 8 (schoolbook) or can it be lower?
    # For 2x2x4, the rank is known to be 8 (I think).
    
    # What about 2x3x2? The rank of 2x3x2 matrix multiplication.
    # Schoolbook: 12. Can it be lower?
    # I believe the rank of 2x3x2 is 12 (no known improvement).
    
    # What about 2x3x4? Schoolbook: 24.
    
    # Let's just validate the current solution and return it.
    candidate = {"U": U, "V": V, "W": W}
    if _validate(candidate):
        return candidate
    
    return _schoolbook()
