<<<<<<< SEARCH
        # Local search to fill gaps
        A_state, B_state = local_search(A_state, B_state, deadline - 1.0, rng)

        total = A_state.size() + B_state.size()
        if total > best_total:
            best_total = total
            best_A = set(A_state.els)
            best_B = set(B_state.els)

        # Try reverse order: B first, then A
=======
        # Local search to fill gaps
        A_state, B_state = local_search(A_state, B_state, deadline - 1.0, rng)

        total = A_state.size() + B_state.size()
        if total > best_total:
            best_total = total
            best_A = set(A_state.els)
            best_B = set(B_state.els)

        # Shake and refill to escape local optima
        if total >= best_total - 1 and time.time() < deadline - 3.0:
            A_state2 = SidonState()
            A_state2.els = set(A_state.els)
            B_state2 = SidonState()
            B_state2.els = set(B_state.els)
            for x in A_state2.els:
                A_state2.sums.add(x + x)
                for y in A_state2.els:
                    if y != x:
                        A_state2.sums.add(x + y)
                        A_state2.diffs.add(x - y)
                        A_state2.diffs.add(y - x)
            for x in B_state2.els:
                B_state2.sums.add(x + x)
                for y in B_state2.els:
                    if y != x:
                        B_state2.sums.add(x + y)
                        B_state2.diffs.add(x - y)
                        B_state2.diffs.add(y - x)
            shake_and_refill(A_state2, B_state2, deadline - 1.0, rng)
            total2 = A_state2.size() + B_state2.size()
            if total2 > best_total:
                best_total = total2
                best_A = set(A_state2.els)
                best_B = set(B_state2.els)

        # Try reverse order: B first, then A
>>>>>>> REPLACE
