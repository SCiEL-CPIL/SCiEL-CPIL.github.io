"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Set of size 19
    # Removed 12 and 20 from {4..24} to make 4 safe
    # Removed 8 from {8..24} to make 8 safe (actually 8 was safe in {9..24}, 
    # but we included 8 and removed 12 for the 4 case? 
    # Let's re-verify the set:
    # S = {4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24}
    # Missing: 1, 2, 3, 12, 20
    return [4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24]
# EVOLVE-BLOCK-END
