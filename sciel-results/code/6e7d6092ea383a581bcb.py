<<<<<<< SEARCH
        # Reset state to best found so far
        A = list(best_A)
        B = list(best_B)
        sums_A = 0
        sums_B = 0
        diffs_A = 0
        diffs_B = 0
        contrib_sums_A = [0] * (N + 1)
        contrib_diffs_A = [0] * (N + 1)
        contrib_sums_B = [0] * (N + 1)
        contrib_diffs_B = [0] * (N + 1)
        for i, a in enumerate(A):
            sums_A |= (1 << (a + a))
            for j, a2 in enumerate(A):
                if j > i:
                    s = a + a2
                    d = abs(a - a2)
                    sums_A |= (1 << s)
                    diffs_A |= (1 << d)
                    contrib_sums_A[a] |= (1 << s)
                    contrib_diffs_A[a] |= (1 << d)
                    contrib_sums_A[a2] |= (1 << s)
                    contrib_diffs_A[a2] |= (1 << d)
        for i, b in enumerate(B):
            sums_B |= (1 << (b + b))
            for j, b2 in enumerate(B):
                if j > i:
                    s = b + b2
                    d = abs(b - b2)
                    sums_B |= (1 << s)
                    diffs_B |= (1 << d)
                    contrib_sums_B[b] |= (1 << s)
                    contrib_diffs_B[b] |= (1 << d)
                    contrib_sums_B[b2] |= (1 << s)
                    contrib_diffs_B[b2] |= (1 << d)
        
        in_A = set(A)
        in_B = set(B)
=======
        # Reset state to best found so far
        A = list(best_A)
        B = list(best_B)
        rebuild_A()
        rebuild_B()
        
        in_A = set(A)
        in_B = set(B)
>>>>>>> REPLACE

