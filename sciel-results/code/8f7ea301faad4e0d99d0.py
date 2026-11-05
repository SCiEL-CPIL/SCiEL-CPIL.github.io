<<<<<<< SEARCH
    def solve_linear_system_exact(A, b):
        """Solve A*x = b over rationals using Gaussian elimination"""
        n_rows = len(A)
        n_cols = len(A[0]) if n_rows > 0 else 0
        n_unknowns = len(b)
        
        # Augmented matrix
        aug = [[A[i][j] for j in range(n_unknowns)] + [b[i]] for i in range(n_rows)]
        
        # Gaussian elimination
        rank = 0
        for col in range(n_unknowns):
            if rank >= n_rows:
                break
            # Find pivot
            pivot_row = None
            for row in range(rank, n_rows):
                if aug[row][col] != 0:
                    pivot_row = row
                    break
            if pivot_row is None:
                continue
            
            # Swap
            aug[rank], aug[pivot_row] = aug[pivot_row], aug[rank]
            
            # Eliminate
            pivot = aug[rank][col]
            for j in range(col, n_unknowns + 1):
                aug[rank][j] = aug[rank][j] / pivot
            
            for row in range(n_rows):
                if row != rank and aug[row][col] != 0:
                    factor = aug[row][col]
                    for j in range(col, n_unknowns + 1):
                        aug[row][j] = aug[row][j] - factor * aug[rank][j]
            
            rank += 1
        
        # Check for inconsistency
        for row in range(rank, n_rows):
            if any(aug[row][j] != 0 for j in range(n_unknowns)):
                return None
        
        # Free variables are set to 0
        x = [Fraction(0)] * n_unknowns
        for row in range(rank):
            # Find the leading column
            for col in range(n_unknowns):
                if aug[row][col] != 0:
                    x[col] = aug[row][n_unknowns]
                    break
        
        return x
=======
    def solve_linear_system_exact(A, b):
        """Solve A*x = b over rationals using Gaussian elimination"""
        n_rows = len(A)
        if n_rows == 0:
            return []
        n_cols = len(A[0])
        
        # Augmented matrix
        aug = [[A[i][j] for j in range(n_cols)] + [b[i]] for i in range(n_rows)]
        
        # Gaussian elimination (Gauss-Jordan to get RREF)
        rank = 0
        for col in range(n_cols):
            if rank >= n_rows:
                break
            # Find pivot
            pivot_row = None
            for row in range(rank, n_rows):
                if aug[row][col] != 0:
                    pivot_row = row
                    break
            if pivot_row is None:
                continue
            
            # Swap
            aug[rank], aug[pivot_row] = aug[pivot_row], aug[rank]
            
            # Eliminate
            pivot = aug[rank][col]
            # Normalize pivot row
            for j in range(col, n_cols + 1):
                aug[rank][j] = aug[rank][j] / pivot
            
            # Eliminate other rows
            for row in range(n_rows):
                if row != rank and aug[row][col] != 0:
                    factor = aug[row][col]
                    for j in range(col, n_cols + 1):
                        aug[row][j] = aug[row][j] - factor * aug[rank][j]
            
            rank += 1
        
        # Check for inconsistency
        for row in range(rank, n_rows):
            if any(aug[row][j] != 0 for j in range(n_cols)):
                return None
        
        # Extract solution
        x = [Fraction(0)] * n_cols
        for row in range(rank):
            # Find the leading column in this row
            for col in range(n_cols):
                if aug[row][col] != 0:
                    x[col] = aug[row][n_cols]
                    break
        
        return x
>>>>>>> REPLACE

The `try_rank_linear_solve` function also needs to be guarded against taking too long per attempt, although the outer loop checks time. The main issue was the crash. The fix above resolves the `IndexError`.

I will also add a small optimization: if `R` is close to schoolbook, the probability of finding a solution with random V/W is low. I will reduce `max_attempts` in the call to `try_rank_linear_solve` to prevent wasting time on unlikely candidates, ensuring the fallback is reached quickly if no solution is found.

