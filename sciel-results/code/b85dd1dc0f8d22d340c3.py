"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 138  # safety margin before 150s kill

    base = 51975

    # All odd divisors of 51975 greater than 1
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    divisors.sort()  # INCREASING order
    num_moduli = len(divisors)

    best_score = base
    best_classes = None

    def greedy_initialize(order, rng):
        """Greedy initialization with a given order of moduli."""
        counts = np.zeros(base, dtype=np.int8)
        residues = [0] * num_moduli

        for idx in order:
            m = divisors[idx]
            best_r = 0
            best_gain = -1
            # For small moduli, check all residues; for large, sample
            if m <= 200:
                for r in range(m):
                    gain = np.count_nonzero(counts[r::m] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
            else:
                # Sample residues
                cands = set(range(min(m, 32)))
                cands.update(rng.randint(0, m - 1) for _ in range(40))
                for r in cands:
                    gain = np.count_nonzero(counts[r::m] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r

            residues[idx] = best_r
            counts[best_r::m] += 1

        score = int(np.count_nonzero(counts == 0))
        return residues, counts, score

    def local_search(residues, counts, score, rng, time_limit):
        """Simulated annealing local search."""
        current_residues = residues[:]
        current_counts = counts.copy()
        current_score = score

        temperature = 10.0
        cooling = 0.9995
        min_temp = 0.5
        best_local_score = current_score
        best_local_residues = current_residues[:]

        start_ls = time.time()

        while time.time() - start_ls < time_limit and temperature > min_temp:
            # Pick a random modulus index
            i = rng.randint(0, num_moduli - 1)
            m = divisors[i]
            old_r = current_residues[i]
            new_r = rng.randint(0, m - 1)

            if new_r == old_r:
                continue

            # Update counts
            current_counts[old_r::m] -= 1
            current_counts[new_r::m] += 1

            new_score = int(np.count_nonzero(current_counts == 0))
            delta = new_score - current_score

            if delta < 0 or rng.random() < np.exp(-delta / temperature):
                current_residues[i] = new_r
                current_score = new_score
                if current_score < best_local_score:
                    best_local_score = current_score
                    best_local_residues = current_residues[:]
            else:
                # Revert
                current_counts[new_r::m] -= 1
                current_counts[old_r::m] += 1

            temperature *= cooling

        return best_local_residues, best_local_score

    def hill_climb(residues, counts, score, rng, time_limit):
        """Deterministic hill climbing: try each modulus in sequence, pick best residue."""
        current_residues = residues[:]
        current_counts = counts.copy()
        current_score = score
        improved = True
        start_hc = time.time()

        while improved and time.time() - start_hc < time_limit:
            improved = False
            for i in range(num_moduli):
                if time.time() - start_hc > time_limit:
                    break
                m = divisors[i]
                old_r = current_residues[i]

                # Compute score without this class
                current_counts[old_r::m] -= 1
                base_uncov = int(np.count_nonzero(current_counts == 0))

                best_r = old_r
                best_uncov = base_uncov

                if m <= 150:
                    for r2 in range(m):
                        if r2 == old_r:
                            continue
                        current_counts[r2::m] += 1
                        unc = int(np.count_nonzero(current_counts == 0))
                        current_counts[r2::m] -= 1
                        if unc < best_uncov:
                            best_uncov = unc
                            best_r = r2
                else:
                    cands = set(range(min(m, 32)))
                    cands.update(rng.randint(0, m - 1) for _ in range(40))
                    for r2 in cands:
                        if r2 == old_r:
                            continue
                        current_counts[r2::m] += 1
                        unc = int(np.count_nonzero(current_counts == 0))
                        current_counts[r2::m] -= 1
                        if unc < best_uncov:
                            best_uncov = unc
                            best_r = r2

                current_counts[old_r::m] += 1

                if best_r != old_r:
                    current_residues[i] = best_r
                    current_counts[old_r::m] -= 1
                    current_counts[best_r::m] += 1
                    current_score = best_uncov
                    improved = True

        return current_residues, current_score

    def remove_redundant(classes):
        """Remove classes that don't contribute to coverage."""
        if not classes:
            return classes
        counts = np.zeros(base, dtype=np.int8)
        for m, r in classes:
            counts[r::m] += 1
        cur_uncov = int(np.count_nonzero(counts == 0))

        new_classes = []
        for cls in classes:
            m, r = cls
            counts[r::m] -= 1
            unc_without = int(np.count_nonzero(counts == 0))
            if unc_without <= cur_uncov:
                # This class is redundant, don't add it back
                pass
            else:
                new_classes.append(cls)
                counts[r::m] += 1
                cur_uncov = unc_without  # update to reflect removal of redundant classes

        return new_classes

    rng = random.Random(42)

    # Multiple restarts with different strategies
    strategies = [
        lambda rng: list(range(num_moduli)),  # natural order (small to large)
        lambda rng: list(range(num_moduli - 1, -1, -1)),  # reverse (large to small)
        lambda rng: rng.sample(range(num_moduli), num_moduli),  # random
    ]

    for restart in range(15):
        if time.time() > deadline - 10:
            break

        strategy_idx = restart % len(strategies)
        order = strategies[strategy_idx](rng)

        residues, counts, score = greedy_initialize(order, rng)

        if score < best_score:
            best_score = score
            best_classes = [[divisors[i], residues[i]] for i in range(num_moduli)]

        # Local search
        if time.time() > deadline - 15:
            break

        ls_time = min(15.0, deadline - 15 - time.time())
        if ls_time < 2.0:
            continue

        best_residues, best_ls_score = local_search(residues, counts, score, rng, ls_time)

        if best_ls_score < best_score:
            best_score = best_ls_score
            best_classes = [[divisors[i], best_residues[i]] for i in range(num_moduli)]

        # Hill climb on the best found
        if time.time() > deadline - 10:
            break

        hc_time = min(10.0, deadline - 10 - time.time())
        if hc_time < 1.0:
            continue

        hc_residues, hc_score = hill_climb(best_residues, counts, best_ls_score, rng, hc_time)

        if hc_score < best_score:
            best_score = hc_score
            best_classes = [[divisors[i], hc_residues[i]] for i in range(num_moduli)]

    # Final polish: remove redundant classes
    if best_classes is not None and time.time() < deadline - 5:
        best_classes = remove_redundant(best_classes)
        # Recompute score
        counts = np.zeros(base, dtype=np.int8)
        for m, r in best_classes:
            counts[r::m] += 1
        best_score = int(np.count_nonzero(counts == 0))

    return best_classes
# EVOLVE-BLOCK-END
