import time
import math

# Fixed inputs
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
LCM_VAL = 30030  # Precomputed LCM of DIVISORS
N = len(DIVISORS)


def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin for 60s limit

    # We want to find distinct multiples v_i of d_i such that max(v)-min(v) is minimized.
    # Iterate L (window length) from 1 upward; for each L, iterate S (window start).
    # For each (L, S), solve the all-different CSP via backtracking with MRV.

    for L in range(1, 100):
        if time.time() > deadline:
            break

        for S in range(LCM_VAL):
            if time.time() > deadline:
                break

            # Build domains: for each divisor, list of multiples in [S, S+L]
            domains = []
            valid = True
            for d in DIVISORS:
                start_mult = ((S + d - 1) // d) * d
                if start_mult > S + L:
                    valid = False
                    break
                candidates = []
                m = start_mult
                while m <= S + L:
                    candidates.append(m)
                    m += d
                domains.append(candidates)

            if not valid:
                continue

            # Quick feasibility: if any domain has size < 1, skip (already handled).
            # Also, if total domain capacity < N, skip.
            total_cap = sum(len(dom) for dom in domains)
            if total_cap < N:
                continue

            # Solve CSP with backtracking + MRV using bitsets for used values.
            # Map each value in [S, S+L] to a bit position 0..L.
            # For each variable, store domain as a bitmask over these L+1 positions.
            # Also store the actual values for reconstruction.
            val_to_bit = {}
            bit_to_val = {}
            for pos, val in enumerate(range(S, S + L + 1)):
                val_to_bit[val] = pos
                bit_to_val[pos] = val

            # Build bitmask domains
            domain_masks = []
            for dom in domains:
                mask = 0
                for v in dom:
                    mask |= (1 << val_to_bit[v])
                domain_masks.append(mask)

            # Precompute variable order by initial domain size (ascending) for MRV initialization.
            var_order = sorted(range(N), key=lambda i: domain_masks[i].bit_count())

            assignment = [-1] * N  # stores bit position assigned, or -1 if unassigned
            used_mask = 0

            def backtrack(depth):
                nonlocal used_mask
                if depth == N:
                    return True

                # MRV: find unassigned variable with fewest available values
                best_var = -1
                best_avail = -1
                best_mask = 0
                for var in var_order:
                    if assignment[var] != -1:
                        continue
                    avail = domain_masks[var] & ~used_mask
                    cnt = avail.bit_count()
                    if cnt == 0:
                        return False
                    if best_var == -1 or cnt < best_avail:
                        best_avail = cnt
                        best_var = var
                        best_mask = avail
                        if cnt == 1:
                            break  # Can't do better than 1

                if best_var == -1:
                    return False

                # Try each available value for best_var
                mask = best_mask
                while mask:
                    low_bit = mask & (-mask)
                    pos = low_bit.bit_length() - 1
                    mask ^= low_bit

                    assignment[best_var] = pos
                    new_used = used_mask | low_bit

                    if backtrack(depth + 1):
                        return True

                    assignment[best_var] = -1
                    used_mask = new_used  # restore (since we OR'd, we need to undo)
                    # Actually, we set used_mask = new_used inside, so we need to restore properly.
                    # Let's use a different approach: pass used_mask as argument.
                    # But for now, let's fix the restore.

                return False

            # The above has a bug in used_mask restoration. Let's rewrite with passed state.
            def backtrack2(depth, used):
                if depth == N:
                    return True

                # MRV
                best_var = -1
                best_avail = -1
                best_mask = 0
                for var in var_order:
                    if assignment[var] != -1:
                        continue
                    avail = domain_masks[var] & ~used
                    cnt = avail.bit_count()
                    if cnt == 0:
                        return False
                    if best_var == -1 or cnt < best_avail:
                        best_avail = cnt
                        best_var = var
                        best_mask = avail
                        if cnt == 1:
                            break

                if best_var == -1:
                    return False

                mask = best_mask
                while mask:
                    low_bit = mask & (-mask)
                    pos = low_bit.bit_length() - 1
                    mask ^= low_bit

                    assignment[best_var] = pos
                    if backtrack2(depth + 1, used | low_bit):
                        return True
                    assignment[best_var] = -1

                return False

            if backtrack2(0, 0):
                # Reconstruct solution
                result = [bit_to_val[assignment[i]] for i in range(N)]
                # Validate
                if (all(result[i] % DIVISORS[i] == 0 for i in range(N))
                        and len(set(result)) == N
                        and all(1 <= x <= MAX_VAL for x in result)):
                    return [int(x) for x in result]

    # Fallback: return divisors themselves
    return DIVISORS[:]

# EVOLVE-BLOCK-END
