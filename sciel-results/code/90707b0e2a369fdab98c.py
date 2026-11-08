import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition block by block.
    # Split n=3 into 2+1, m=5 into 3+2, p=5 into 3+2.
    # This creates 8 blocks of size (n_i, m_j, p_k).
    # n_i in {2, 1}, m_j in {3, 2}, p_k in {3, 2}.
    
    # Ranks for each block:
    # (2, 3, 3): Schoolbook 18
    # (2, 3, 2): Schoolbook 12
    # (2, 2, 3): Schoolbook 12
    # (2, 2, 2): Strassen 7
    # (1, 3, 3): Schoolbook 9
    # (1, 3, 2): Schoolbook 6
    # (1, 2, 3): Schoolbook 6
    # (1, 2, 2): Schoolbook 4
    
    # Total Rank: 18 + 12 + 12 + 7 + 9 + 6 + 6 + 4 = 74.
    # Wait, 2x2x2 is 7.
    # 18+12+12+7 = 49.
    # 9+6+6+4 = 25.
    # Total 74.
    # This is only 1 better than 75.
    
    # Can we do better for 2x3x3?
    # 2x3x3 is 18.
    # Maybe use Strassen for 2x2x2 inside?
    # 2x3x3 = 2x2x3 + 2x1x3.
    # 2x2x3 = 2x2x2 + 2x2x1.
    # 2x2x2 = 7.
    # 2x2x1 = 4.
    # 2x1x3 = 6.
    # Total 7+4+6 = 17.
    # So 2x3x3 can be 17.
    
    # Let's recalculate with optimized blocks:
    # (2, 3, 3): 17
    # (2, 3, 2): 2x2x2(7) + 2x1x2(4) + 2x2x1(4) + 2x1x1(2) = 17?
    # 2x3x2 = 2x2x2 + 2x1x2 + 2x2x1 + 2x1x1.
    # 7 + 4 + 4 + 2 = 17.
    # (2, 2, 3): 2x2x2(7) + 2x2x1(4) = 11.
    # (2, 2, 2): 7.
    # (1, 3, 3): 9.
    # (1, 3, 2): 6.
    # (1, 2, 3): 6.
    # (1, 2, 2): 4.
    
    # Total: 17 + 17 + 11 + 7 + 9 + 6 + 6 + 4 = 77.
    # This is worse.
    # The overhead of splitting increases rank for small blocks.
    
    # Stick to the 74 estimate or try to find a global structure.
    # For now, implement the 74 rank decomposition.
    
    pass
