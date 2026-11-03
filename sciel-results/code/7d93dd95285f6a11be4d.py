# --- Part 2: 4x4x2 part (B cols 4-5, C cols 4-5) ---
    # Use trivial rank 4 decomposition for each 2x2x2 block multiplication
    # Four multiplications:
    # 1. A00 * B0 -> C0
    # 2. A01 * B1 -> C0
    # 3. A10 * B0 -> C1
    # 4. A11 * B1 -> C1
    
    mults = [
