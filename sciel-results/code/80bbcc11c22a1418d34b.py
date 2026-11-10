"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from math import gcd
    from fractions import Fraction

    deadline = time.time() + 110  # safety margin before 120s limit

    # Precompute weights: weight[a] = 1/(512-a) as Fraction
    N = 512
    weights = {}
    for a in range(1, N):
        weights[a] = Fraction(1, N - a)

    def score_of(chosen):
        s = Fraction(0, 1)
        for a in chosen:
            s += weights[a]
        return s

    def greedy_init():
        """Greedy: pick largest a first, skip if not coprime with all chosen."""
        chosen = []
        chosen_set = set()
        for a in range(511, 0, -1):
            if all(gcd(a, b) == 1 for b in chosen):
                chosen.append(a)
                chosen_set.add(a)
        return chosen, chosen_set

    def greedy_after_removal(chosen, removed):
        """Re-run greedy excluding 'removed' from consideration."""
        chosen_set = set(chosen) - {removed}
        new_chosen = []
        # Sort remaining by descending value (largest a first)
        remaining = [a for a in range(511, 0, -1) if a != removed]
        for a in remaining:
            if all(gcd(a, b) == 1 for b in new_chosen):
                new_chosen.append(a)
        return new_chosen

    # Initial greedy
    best, best_set = greedy_init()
    best_score = score_of(best)
    best_list = list(best)

    # Local search: try removing each element and re-greedy
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best_list)):
            if time.time() > deadline:
                break
            removed = best_list[i]
            # Re-greedy without the removed element
            candidate = greedy_after_removal(best_list, removed)
            cand_score = score_of(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_list = candidate
                best = candidate
                improved = True
                break  # restart scan with new best

    return best_list
# EVOLVE-BLOCK-END
