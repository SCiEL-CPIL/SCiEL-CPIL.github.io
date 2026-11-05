def _strassen_4x4x4():
    """Return U,V,W for 4x4x4 matrix multiplication using Strassen (rank 49)."""
    # Strassen 2x2x2 has rank 7
    # 4x4x4 = 7 subproblems of 2x2x2, each rank 7, total 49
    # ...
