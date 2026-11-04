"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s limit to be safe

    n, m, p = DIMENSIONS

    # Start with the known 6-block Strassen packing (rank 99)
    blocks = [
        (0, 0, 0), (0, 2, 0), (0, 0, 2),
        (0, 2, 2), (0, 0, 4), (0, 2, 4)
    ]

    def build_decomposition(blocks, remaining_terms):
        """Build U, V, W factor matrices from Strassen blocks and schoolbook terms."""
        rank = len(blocks) * 7 + len(remaining_terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0
        for (i0, j0, k0) in blocks:
            # M1 = (A00+A11)(B00+B11)
            U[i0*m+j0][col] += 1; U[(i0+1)*m+(j0+1)][col] += 1
            V[j0*p+k0][col] += 1; V[(j0+1)*p+(k0+1)][col] += 1
            W[k0*n+i0][col] += 1; W[(k0+1)*n+(i0+1)][col] += 1
            col += 1
            # M2 = (A10+A11)*B00
            U[(i0+1)*m+j0][col] += 1; U[(i0+1)*m+(j0+1)][col] += 1
            V[j0*p+k0][col] += 1
            W[k0*n+(i0+1)][col] += 1; W[(k0+1)*n+(i0+1)][col] -= 1
            col += 1
            # M3 = A00*(B10-B11)
            U[i0*m+j0][col] += 1
            V[j0*p+(k0+1)][col] += 1; V[(j0+1)*p+(k0+1)][col] -= 1
            W[(k0+1)*n+i0][col] += 1; W[(k0+1)*n+(i0+1)][col] += 1
            col += 1
            # M4 = A11*(B01-B00)
            U[(i0+1)*m+(j0+1)][col] += 1
            V[(j0+1)*p+k0][col] += 1; V[j0*p+k0][col] -= 1
            W[k0*n+i0][col] += 1; W[k0*n+(i0+1)][col] += 1
            col += 1
            # M5 = (A00+A01)*B11
            U[i0*m+j0][col] += 1; U[i0*m+(j0+1)][col] += 1
            V[(j0+1)*p+(k0+1)][col] += 1
            W[(k0+1)*n+i0][col] += 1; W[k0*n+i0][col] -= 1
            col += 1
            # M6 = (A10-A00)*(B00+B10)
            U[(i0+1)*m+j0][col] += 1; U[i0*m+j0][col] -= 1
            V[j0*p+k0][col] += 1; V[j0*p+(k0+1)][col] += 1
            W[(k0+1)*n+(i0+1)][col] += 1
            col += 1
            # M7 = (A01-A11)*(B01+B11)
            U[i0*m+(j0+1)][col] += 1; U[(i0+1)*m+(j0+1)][col] -= 1
            V[(j0+1)*p+k0][col] += 1; V[(j0+1)*p+(k0+1)][col] += 1
            W[k0*n+i0][col] += 1
            col += 1

        for (i, j, k) in remaining_terms:
            U[i*m+j][col] = 1
            V[j*p+k][col] = 1
            W[k*n+i][col] = 1
            col += 1

        return U, V, W

    def verify(U, V, W):
        """Verify that U @ diag @ V^T reconstructs the tensor via W."""
        rank = len(U[0])
        # Check: sum_r U[i*m+j][r] * V[j*p+k][r] == W[k*n+i][r] for all r, summed
        # Actually the decomposition means: T[i,j,k] = sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        # So for each (i,j,k), sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True

    def get_covered_terms(blocks):
        covered = set()
        for (i0, j0, k0) in blocks:
            for i in (i0, i0+1):
                for j in (j0, j0+1):
                    for k in (k0, k0+1):
                        covered.add((i, j, k))
        return covered

    def get_remaining_terms(blocks):
        covered = get_covered_terms(blocks)
        return [(i, j, k) for i in range(n) for j in range(m) for k in range(p)
                if (i, j, k) not in covered]

    # Initial decomposition
    remaining = get_remaining_terms(blocks)
    U, V, W = build_decomposition(blocks, remaining)
    best_rank = len(blocks) * 7 + len(remaining)
    best_result = (U, V, W)

    # Try to reduce rank by finding linear dependencies among columns
    # Use Gaussian elimination over rationals to find a basis of the column space
    def try_reduce_rank(U, V, W, current_rank, max_time):
        """Try to reduce rank by finding linear dependencies."""
        start = time.time()
        
        # Build the tensor T from U, V, W
        # T[i,j,k] = sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        # We want to find if some columns are linearly dependent in a way that preserves the tensor
        
        # Approach: try to express each column as a linear combination of others
        # This is equivalent to checking if the rank of the "column tensor" is less than current_rank
        
        # A simpler approach: use the fact that if two columns have the same
        # "pattern" up to scalar, we can merge them
        # Or: use SVD-like approach on the unfolded tensor
        
        # For now, try a greedy approach: check if removing column r and adjusting
        # other columns can preserve the tensor
        
        # Actually, let's try a different approach: use the LLL algorithm or
        # simply check for small integer relations
        
        # For this specific problem, let's try to see if we can do better than 99
        # by using a different block packing or by combining blocks
        
        # Try all possible sets of disjoint 2x2x2 blocks
        all_blocks = []
        for i0 in range(n-1):
            for j0 in range(m-1):
                for k0 in range(p-1):
                    all_blocks.append((i0, j0, k0))
        
        # Find maximum set of disjoint blocks
        def find_max_disjoint_blocks(block_list, deadline):
            best_count = 0
            best_set = []
            
            def backtrack(remaining_blocks, current, depth):
                nonlocal best_count, best_set
                if time.time() > deadline:
                    return
                if len(current) > best_count:
                    best_count = len(current)
                    best_set = list(current)
                if not remaining_blocks:
                    return
                for idx, blk in enumerate(remaining_blocks):
                    # Check if blk is disjoint from all blocks in current
                    disjoint = True
                    for cblk in current:
                        if (abs(blk[0]-cblk[0]) < 2 and abs(blk[1]-cblk[1]) < 2 and abs(blk[2]-cblk[2]) < 2):
                            disjoint = False
                            break
                    if disjoint:
                        new_remaining = remaining_blocks[:idx] + remaining_blocks[idx+1:]
                        backtrack(new_remaining, current + [blk], depth + 1)
            
            backtrack(block_list, [], 0)
            return best_set
        
        max_blocks = find_max_disjoint_blocks(all_blocks, deadline - 10)
        
        if len(max_blocks) > len(blocks):
            new_remaining = get_remaining_terms(max_blocks)
            new_U, new_V, new_W = build_decomposition(max_blocks, new_remaining)
            new_rank = len(max_blocks) * 7 + len(new_remaining)
            if new_rank < current_rank and verify(new_U, new_V, new_W):
                return new_U, new_V, new_W, new_rank
        
        return None, None, None, current_rank

    # Try to reduce rank
    if time.time() < deadline - 20:
        new_U, new_V, new_W, new_rank = try_reduce_rank(U, V, W, best_rank, deadline - 20)
        if new_U is not None and new_rank < best_rank:
            best_rank = new_rank
            best_result = (new_U, new_V, new_W)

    # Final verification
    U, V, W = best_result
    if not verify(U, V, W):
        # Fallback to known good solution
        blocks = [
            (0, 0, 0), (0, 2, 0), (0, 0, 2),
            (0, 2, 2), (0, 0, 4), (0, 2, 4)
        ]
        remaining = get_remaining_terms(blocks)
        U, V, W = build_decomposition(blocks, remaining)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
