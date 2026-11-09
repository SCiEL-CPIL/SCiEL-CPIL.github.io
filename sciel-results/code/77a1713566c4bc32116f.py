"""Program to find a rank-minimal exact tensor decomposition for 3x5 by 5x7 matrix multiplication."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Start with schoolbook decomposition (rank 105)
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    def verify(U, V, W, n, m, p):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1) > 1e-10:
                        return False
        return True
    
    # Strategy: Use randomized algebraic search over rationals to find lower rank decompositions
    # We model the tensor as a bilinear map and try to find rank-1 terms that sum to the identity
    
    def try_random_decomposition(target_rank, n, m, p, max_iters=1000, deadline=None):
        """
        Try to find a decomposition of rank target_rank using random initialization and iterative refinement.
        Returns (U, V, W) if found, None otherwise.
        """
        if deadline is not None and time.time() > deadline:
            return None
        
        # The tensor T_{i,j,k} = delta_{i,j,k} for matrix multiplication A*B = C
        # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        # and 0 otherwise? No, the tensor is 1 only for the specific (i,j,k) triples that form the product.
        # Actually, the matrix multiplication tensor T has T_{i,j,k} = 1 if the entry (i,k) of AB depends on A_{i,j} and B_{j,k}.
        # For standard matrix multiplication, C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # So the tensor T_{i,j,k} is 1 for all i in [0,n), j in [0,m), k in [0,p).
        # Wait, the index convention is W[k*n+i, r].
        # The tensor is T_{i,j,k} = 1 for all valid i,j,k.
        # We need sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = 1 for all i,j,k.
        
        # This is a system of n*m*p equations in 3*rank variables.
        # Let's try a random approach:
        # Initialize U, V, W with small random rational numbers.
        # Use gradient descent or a specialized solver.
        
        # Given the complexity, let's try a different approach:
        # Greedy rank-1 peeling.
        # Start with the target tensor T (all ones in the relevant subspace).
        # Repeatedly find a rank-1 tensor that best approximates the residual, subtract it, repeat.
        
        # However, exact rational arithmetic is required.
        # Let's try to solve a linear system for the last term if we fix the first rank-1 terms.
        
        # A simpler heuristic:
        # Try to find a decomposition where U, V, W have sparse entries.
        
        # For now, this function is a placeholder for more advanced search.
        # We will implement a simple random search for small ranks.
        
        return None

    # Try to find lower rank decompositions
    # Known bounds: rank(M_{n,m,p}) <= n*m*p
    # For 3x5x7, 105 is the schoolbook rank.
    # Can we do better?
    # Strassen's algorithm reduces rank for 2x2x2 from 8 to 7.
    # For larger dimensions, block-wise Strassen can help.
    
    # Let's try to use a block decomposition.
    # If we can decompose the 3x5x7 problem into smaller blocks, we might reduce the rank.
    
    # Example: Split the 5x5 middle dimension into 2+3 or 3+2.
    # A (3x5) * B (5x7)
    # Split B into B1 (5x3) and B2 (5x4)? No, split the inner dimension.
    # A = [A1 (3x3), A2 (3x2)]
    # B = [B1; B2] where B1 is 3x7 and B2 is 2x7
    # C = A1*B1 + A2*B2
    # Rank(C) <= Rank(A1*B1) + Rank(A2*B2)
    # Rank(A1*B1) for 3x3x3 is at most 27 (schoolbook) or 19 (Strassen-like?)
    # Rank(A2*B2) for 3x2x7 is at most 42 (schoolbook)
    # Total: 27 + 42 = 69 < 105.
    
    # Let's implement this block decomposition.
    
    def block_decompose(n1, m1, p1, n2, m2, p2):
        """
        Decompose matrix multiplication of two blocks.
        Block 1: n1 x m1 * m1 x p1
        Block 2: n2 x m2 * m2 x p2
        Returns U, V, W for the combined system.
        """
        rank1 = n1 * m1 * p1
        rank2 = n2 * m2 * p2
        total_rank = rank1 + rank2
        
        U = [[0] * total_rank for _ in range(n1 * m1 + n2 * m2)]
        V = [[0] * total_rank for _ in range(m1 * p1 + m2 * p2)]
        W = [[0] * total_rank for _ in range(p1 * n1 + p2 * n2)]
        
        # Fill Block 1
        col = 0
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    U[i * m1 + j][col] = 1
                    V[j * p1 + k][col] = 1
                    W[k * n1 + i][col] = 1
                    col += 1
        
        # Fill Block 2
        # Note: The indices for Block 2 are offset in the global matrices.
        # U rows for Block 2 start at n1 * m1
        # V rows for Block 2 start at m1 * p1
        # W rows for Block 2 start at p1 * n1
        
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    U[(n1 * m1) + i * m2 + j][col] = 1
                    V[(m1 * p1) + j * p2 + k][col] = 1
                    W[(p1 * n1) + k * n2 + i][col] = 1
                    col += 1
        
        return U, V, W

    # Try splitting the inner dimension m=5 into 2 and 3
    # A is 3x5, B is 5x7
    # Split A into A1 (3x2) and A2 (3x3)
    # Split B into B1 (2x7) and B2 (3x7)
    # C = A1*B1 + A2*B2
    
    n1, m1, p1 = 3, 2, 7
    n2, m2, p2 = 3, 3, 7
    
    # Check if dimensions match
    # A1: 3x2, B1: 2x7 -> C1: 3x7
    # A2: 3x3, B2: 3x7 -> C2: 3x7
    # Total C: 3x7
    
    U_block, V_block, W_block = block_decompose(n1, m1, p1, n2, m2, p2)
    
    # Verify the block decomposition
    # The global indices are:
    # U: 3*2 + 3*3 = 6 + 9 = 15 rows. Wait, n*m = 3*5 = 15. Correct.
    # V: 2*7 + 3*7 = 14 + 21 = 35 rows. Wait, m*p = 5*7 = 35. Correct.
    # W: 7*3 + 7*3 = 21 + 21 = 42 rows. Wait, p*n = 7*3 = 21. 
    # Ah, W shape is (p*n, rank).
    # In block 1, W rows are p1*n1 = 7*3 = 21.
    # In block 2, W rows are p2*n2 = 7*3 = 21.
    # Total W rows = 42. But we only need 21 rows for the final C (3x7).
    # The block decomposition adds the contributions.
    # C_{i,k} = C1_{i,k} + C2_{i,k}
    # So W should be the sum of W1 and W2 for the same (k,i) indices.
    # My block_decompose function creates a W with 42 rows, but we only have 21 distinct (k,i) pairs.
    # We need to merge the W matrices.
    
    # Let's re-implement the combination properly.
    
    def combine_blocks(n1, m1, p1, n2, m2, p2):
        """
        Combine two matrix multiplication blocks.
        Block 1: n1 x m1 * m1 x p1
        Block 2: n2 x m2 * m2 x p2
        Assumes n1 == n2 and p1 == p2 for direct addition of C.
        """
        if n1 != n2 or p1 != p2:
            return None
        
        rank1 = n1 * m1 * p1
        rank2 = n2 * m2 * p2
        total_rank = rank1 + rank2
        n = n1
        p = p1
        
        U = [[0] * total_rank for _ in range(n * (m1 + m2))]
        V = [[0] * total_rank for _ in range((m1 + m2) * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # Fill Block 1
        col = 0
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    # U index: i * (m1+m2) + j
                    # V index: j * p + k
                    # W index: k * n + i
                    U[i * (m1 + m2) + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        # Fill Block 2
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    # U index: i * (m1+m2) + (m1 + j)
                    # V index: (m1 + j) * p + k
                    # W index: k * n + i
                    U[i * (m1 + m2) + (m1 + j)][col] = 1
                    V[(m1 + j) * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        return U, V, W

    # Try split 5 = 2 + 3
    U_split, V_split, W_split = combine_blocks(3, 2, 7, 3, 3, 7)
    if U_split is not None:
        rank_split = len(U_split[0])
        if verify(U_split, V_split, W_split, n, m, p) and rank_split < best_rank:
            best_result = {"U": U_split, "V": V_split, "W": W_split}
            best_rank = rank_split
    
    # Try split 5 = 1 + 4
    U_split, V_split, W_split = combine_blocks(3, 1, 7, 3, 4, 7)
    if U_split is not None:
        rank_split = len(U_split[0])
        if verify(U_split, V_split, W_split, n, m, p) and rank_split < best_rank:
            best_result = {"U": U_split, "V": V_split, "W": W_split}
            best_rank = rank_split

    # Try split 5 = 3 + 2 (same as 2+3 but swapped)
    U_split, V_split, W_split = combine_blocks(3, 3, 7, 3, 2, 7)
    if U_split is not None:
        rank_split = len(U_split[0])
        if verify(U_split, V_split, W_split, n, m, p) and rank_split < best_rank:
            best_result = {"U": U_split, "V": V_split, "W": W_split}
            best_rank = rank_split

    # Can we apply Strassen to the 3x3x3 block?
    # Strassen reduces 2x2x2 from 8 to 7.
    # For 3x3x3, we can embed 2x2x2 blocks.
    # But implementing full Strassen for arbitrary dimensions is complex.
    # Let's stick to the block decomposition for now.
    
    # The split 2+3 gives rank 3*2*7 + 3*3*7 = 42 + 63 = 105. No improvement.
    # The split 1+4 gives rank 3*1*7 + 3*4*7 = 21 + 84 = 105. No improvement.
    # Block decomposition with schoolbook sub-blocks doesn't help.
    
    # We need a better algorithm for the sub-blocks.
    # For 3x3x3, the rank is at most 27.
    # Is there a known lower rank?
    # The rank of M_{3,3,3} is at most 23 (Strassen-like?).
    # If we can reduce the 3x3x3 block, we might improve.
    
    # Let's try to find a lower rank decomposition for 3x3x3.
    # This is a smaller problem.
    
    def solve_3x3x3():
        """Find a rank-minimal decomposition for 3x3x3 matrix multiplication."""
        n, m, p = 3, 3, 3
        rank_schoolbook = n * m * p
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        # Try to find lower rank
        # For 2x2x2, Strassen gives rank 7.
        # For 3x3x3, we can use a 2x2x2 block and 1x1x1 blocks.
        # Or use a known decomposition.
        
        # Let's try a random search for rank < 27.
        # This is computationally expensive.
        
        return U, V, W, rank_schoolbook

    U_333, V_333, W_333, rank_333 = solve_333()
    
    # If we found a better 3x3x3 decomposition, we can use it in the 3x3x7 block.
    # But combining them is non-trivial.
    
    # For now, return the best result found.
    return best_result
