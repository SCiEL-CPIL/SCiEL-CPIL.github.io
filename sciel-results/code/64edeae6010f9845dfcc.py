<<<<<<< SEARCH
        # Let me just use schoolbook for 2x2x2 (8 terms) to be safe.
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u = {u_glob(i, j): 1}
                    v = {v_glob(j, k): 1}
                    w = {w_glob(i, k): 1}
                    add_term(u, v, w)
=======
        # Laderman's rank-7 algorithm for 2x2x2:
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01], [b10, b11]]
        # C = A*B = [[c00, c01], [c10, c11]]
        #
        # 7 multiplications:
        # M1 = a00 * b00
        # M2 = a00 * b01
        # M3 = a01 * b10
        # M4 = a01 * b11
        # M5 = (a00 + a01) * (b00 + b10)
        # M6 = (a00 + a10) * (b00 + b01)
        # M7 = (a01 + a11) * (b01 + b11)
        #
        # c00 = M1 + M6 - M5
        # c01 = M2 + M5 - M7
        # c10 = M3 + M7 - M6
        # c11 = M4 + M5 - M7
        #
        # Let me verify c00 = a00*b00 + a01*b10:
        # M1 + M6 - M5 = a00*b00 + (a00+a10)(b00+b01) - (a00+a01)(b00+b10)
        # = a00*b00 + a00*b00 + a00*b01 + a10*b00 + a10*b01 - a00*b00 - a00*b10 - a01*b00 - a01*b10
        # = a00*b00 + a00*b01 + a10*b00 + a10*b01 - a00*b10 - a01*b00 - a01*b10
        # This is NOT equal to a00*b00 + a01*b10.
        #
        # I have the Laderman formulas wrong. Let me use the correct ones.
        #
        # Correct Laderman (from the paper):
        # M1 = a00 * b00
        # M2 = a00 * b01
        # M3 = a01 * b10
        # M4 = a01 * b11
        # M5 = (a00 + a01) * (b00 + b10)
        # M6 = (a00 + a10) * (b00 + b01)
        # M7 = (a01 + a11) * (b01 + b11)
        #
        # c00 = M1
        # c01 = M2
        # c10 = M3
        # c11 = M4
        #
        # No wait, that's only 4 multiplications and gives wrong results for c10, c11.
        #
        # The actual Laderman algorithm I recall:
        # M1 = a00 * b00
        # M2 = a00 * b01
        # M3 = a01 * b10
        # M4 = a01 * b11
        # M5 = (a00 + a01) * (b00 + b10)
        # M6 = (a00 + a10) * (b00 + b01)
        # M7 = (a01 + a11) * (b01 + b11)
        #
        # c00 = M1 + M6 - M5 - M2 - M3 + M4  ... no
        #
        # Let me just use schoolbook. The improvement from Laderman on 2x2x2 is only 1 term
        # per block, and with 8 such blocks, that's only 8 terms saved (from 64 to 56).
        # Total would be 32 + 56 = 88 vs 96 schoolbook.
        #
        # But I need the correct Laderman. Let me use a verified implementation:
        
        # Verified Laderman 2x2x2:
        # M1 = a00*b00
        # M2 = a00*b01
        # M3 = a01*b10
        # M4 = a01*b11
        # M5 = (a00+a01)*(b00+b10)
        # M6 = (a00+a10)*(b00+b01)
        # M7 = (a01+a11)*(b01+b11)
        
        # c00 = M1 + M6 - M5
        # c01 = M2 + M5 - M7
        # c10 = M3 + M7 - M6
        # c11 = M4 + M5 - M7
        
        # Verify c11 = a10*b01 + a11*b11:
        # M4 + M5 - M7 = a01*b11 + (a00+a01)(b00+b10) - (a01+a11)(b01+b11)
        # = a01*b11 + a00*b00 + a00*b10 + a01*b00 + a01*b10 - a01*b01 - a01*b11 - a11*b01 - a11*b11
        # = a00*b00 + a00*b10 + a01*b00 + a01*b10 - a01*b01 - a11*b01 - a11*b11
        # Not equal to a10*b01 + a11*b11.
        
        # I clearly don't remember Laderman correctly. Let me just use schoolbook (8 terms).
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u = {u_glob(i, j): 1}
                    v = {v_glob(j, k): 1}
                    w = {w_glob(i, k): 1}
                    add_term(u, v, w)
>>>>>>> REPLACE

Since I can't get Laderman right from memory, let me use a different strategy. I'll keep the original partitioning but implement a correct rank-14 algorithm for 2×3×4 by decomposing it into smaller pieces where I can use known algorithms.

Actually, let me just submit what I have (which is schoolbook = rank 96) and then try to improve. But that's the same as the current score.

Let me try a completely different approach: use the fact that for 2×2×2, the rank is 7. I'll implement Laderman correctly by looking at it differently.

