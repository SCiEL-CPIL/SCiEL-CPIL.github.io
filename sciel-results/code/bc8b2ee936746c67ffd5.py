import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 90s limit

    N = 256
    best_size = 0
    best_A = []
    best_B = []

    def compute_masks(S):
        """Compute sum and diff bitmasks for set S."""
        sums = 0
        diffs = 0
        for i in range(len(S)):
            sums |= (1 << (2 * S[i]))
            for j in range(i):
                sums |= (1 << (S[i] + S[j]))
                d = abs(S[i] - S[j])
                if d > 0:
                    diffs |= (1 << d)
        return sums, diffs

    def can_add_to_A(x, A, sums_A, diffs_B):
        """Check if x can be added to A. Returns (ok, new_sums, new_diffs)."""
        # Check sum conflicts within A
        for a in A:
            s = x + a
            if (sums_A >> s) & 1:
                return (False, 0, 0)
        if (sums_A >> (2 * x)) & 1:
            return (False, 0, 0)
        # Build new sums
        ns = sums_A
        for a in A:
            ns |= (1 << (x + a))
        ns |= (1 << (2 * x))
        # Check diff conflicts with B's diffs
        nd = 0
        for a in A:
            d = abs(x - a)
            if d > 0:
                if (diffs_B >> d) & 1:
                    return (False, 0, 0)
                nd |= (1 << d)
        return (True, ns, nd)

    def can_add_to_B(x, B, sums_B, diffs_A):
        """Check if x can be added to B. Returns (ok, new_sums, new_diffs)."""
        for b in B:
            s = x + b
            if (sums_B >> s) & 1:
                return (False, 0, 0)
        if (sums_B >> (2 * x)) & 1:
            return (False, 0, 0)
        ns = sums_B
        for b in B:
            ns |= (1 << (x + b))
        ns |= (1 << (2 * x))
        nd = 0
        for b in B:
            d = abs(x - b)
            if d > 0:
                if (diffs_A >> d) & 1:
                    return (False, 0, 0)
                nd |= (1 << d)
        return (True, ns, nd)

    def greedy_pass(seed_offset=0):
        """Run one greedy pass with random ordering."""
        rng = random.Random(seed_offset)
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        A = []
        B = []
        sums_A = 0
        sums_B = 0
        diffs_A = 0
        diffs_B = 0
        
        for x in order:
            ok, ns, nd = can_add_to_A(x, A, sums_A, diffs_B)
            if ok:
                A.append(x)
                sums_A = ns
                diffs_A |= nd
            else:
                ok2, ns2, nd2 = can_add_to_B(x, B, sums_B, diffs_A)
                if ok2:
                    B.append(x)
                    sums_B = ns2
                    diffs_B |= nd2
        
        return A, B

    def local_search(A, B, time_limit):
        """Fast local search: insertions, moves, swaps using O(k) checks."""
        A = list(A)
        B = list(B)
        polish_deadline = time.time() + time_limit
        improved = True
        no_improve_count = 0
        
        while improved and time.time() < polish_deadline:
            improved = False
            no_improve_count += 1
            
            sums_A, diffs_A = compute_masks(A)
            sums_B, diffs_B = compute_masks(B)
            all_elements = set(A) | set(B)
            available = [x for x in range(1, N + 1) if x not in all_elements]
            rng = random.Random()
            
            # Try insertions (most valuable, +1 each)
            for x in rng.sample(available, min(len(available), 50)):
                ok, ns, nd = can_add_to_A(x, A, sums_A, diffs_B)
                if ok:
                    A.append(x)
                    sums_A = ns
                    diffs_A |= nd
                    improved = True
                    break
                ok2, ns2, nd2 = can_add_to_B(x, B, sums_B, diffs_A)
                if ok2:
                    B.append(x)
                    sums_B = ns2
                    diffs_B |= nd2
                    improved = True
                    break
            if improved:
                continue
            
            # Try moving elements between sets (0 net gain but may enable future inserts)
            moved = False
            for i in rng.sample(range(len(A)), min(len(A), 20)):
                y = A[i]
                A_test = A[:i] + A[i+1:]
                sums_A_test, diffs_A_test = compute_masks(A_test)
                ok, ns, nd = can_add_to_B(y, B, sums_B, diffs_A_test)
                if ok:
                    A.pop(i)
                    B.append(y)
                    moved = True
                    break
            if not moved:
                for i in rng.sample(range(len(B)), min(len(B), 20)):
                    y = B[i]
                    B_test = B[:i] + B[i+1:]
                    sums_B_test, diffs_B_test = compute_masks(B_test)
                    ok, ns, nd = can_add_to_A(y, A, sums_A, diffs_B_test)
                    if ok:
                        B.pop(i)
                        A.append(y)
                        moved = True
                        break
            if moved:
                improved = True
                continue
            
            # Try swaps (remove y from A, add x to A) - can free up diff space
            swapped = False
            for x in rng.sample(available, min(len(available), 30)):
                for i in rng.sample(range(len(A)), min(len(A), 10)):
                    y = A[i]
                    A_test = A[:i] + A[i+1:]
                    sums_A_test, diffs_A_test = compute_masks(A_test)
                    ok, ns, nd = can_add_to_A(x, A_test, sums_A_test, diffs_B)
                    if ok:
                        A[i] = x
                        swapped = True
                        break
                if swapped:
                    break
            if not swapped:
                for x in rng.sample(available, min(len(available), 30)):
                    for i in rng.sample(range(len(B)), min(len(B), 10)):
                        y = B[i]
                        B_test = B[:i] + B[i+1:]
                        sums_B_test, diffs_B_test = compute_masks(B_test)
                        ok, ns, nd = can_add_to_B(x, B_test, sums_B_test, diffs_A)
                        if ok:
                            B[i] = x
                            swapped = True
                            break
                    if swapped:
                        break
            if swapped:
                improved = True
                continue
            
            # If no improvement for several rounds, do a random restart of polish
            if no_improve_count > 15:
                break
        
        return A, B

    # Main loop: greedy passes
    restart_count = 0
    while time.time() < deadline:
        A, B = greedy_pass(restart_count)
        restart_count += 1
        
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = list(A)
            best_B = list(B)
        
        # Also try local search on this result if promising
        if current_size >= best_size - 2 and time.time() < deadline - 5:
            ls_time = min(3.0, max(0.5, deadline - time.time() - 5))
            ls_A, ls_B = local_search(A, B, ls_time)
            ls_size = len(ls_A) + len(ls_B)
            if ls_size > best_size:
                best_size = ls_size
                best_A = list(ls_A)
                best_B = list(ls_B)
    
    # Final polish on best solution
    if best_size > 0 and time.time() < deadline - 2:
        polish_time = min(15.0, max(1.0, deadline - time.time() - 2))
        polished_A, polished_B = local_search(best_A, best_B, polish_time)
        polished_size = len(polished_A) + len(polished_B)
        if polished_size > best_size:
            best_size = polished_size
            best_A = polished_A
            best_B = polished_B

    # Ensure non-empty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
