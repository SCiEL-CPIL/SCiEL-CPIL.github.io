"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell.
    Uses a centered square dielectric rod of size 16x16 (fill factor 0.25).
    This structure is invariant under the square symmetry group (D4).
    """
    size = problem["pixels"]
    # Ensure size is 32 as per problem statement, but handle generally if possible
    if size != 32:
        # Fallback to simple circle if size is not 32, though contract says 32
        cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            row = []
            for j in range(size):
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < 0.2 ** 2:
                    row.append(1)
                else:
                    row.append(0)
            cell.append(row)
        return cell

    # Construct a 16x16 square rod centered in the 32x32 cell
    # The cell is symmetric: C[i][j] == C[31-i][31-j] etc.
    # A 16x16 square centered at (15.5, 15.5) spans indices 8 to 23.
    # Indices 8..23 inclusive is 16 pixels.
    start = size // 2 - 8  # 16 - 8 = 8
    end = size // 2 + 8    # 16 + 8 = 24 (exclusive)
    
    cell = [[0 for _ in range(size)] for _ in range(size)]
    
    for i in range(start, end):
        for j in range(start, end):
            cell[i][j] = 1
            
    return cell
# EVOLVE-BLOCK-END
