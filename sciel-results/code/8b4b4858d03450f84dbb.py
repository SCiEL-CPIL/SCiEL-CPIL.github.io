import time
import random
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    MAX_M = 128

    # Moduli in [2, 128]
    moduli = list(range(2, MAX_M + 1))

    # Two residue classes (r1, m1) and (r2, m2) are disjoint iff gcd(m1, m2) does NOT divide (r1 - r2)
    # i.e., gcd(m1,m2) | (r1-r2)  =>  they intersect
    # We want gcd(m1,m2) ∤ (r1-r2) for all pairs.

    # Strategy: Greedy + local search / randomized search to maximize count.
    # For each modulus, we can choose any residue 0..m-1.
    # We want to select as many moduli as possible and assign residues such that all pairs are disjoint.

    # Key insight: If we pick moduli that are pairwise coprime, then gcd=1 always, and 1 divides everything,
    # so they would ALWAYS intersect. So we need moduli that are NOT coprime.
    # Actually wait: if gcd(m1,m2)=1, then the classes always intersect (CRT). So we need gcd(m1,m2) > 1
    # AND we need to choose residues so that gcd(m1,m2) does not divide (r1-r2).

    # So we want to select a subset of moduli and assign residues to maximize size.
    # This is a combinatorial problem. Let's use a greedy approach with backtracking or randomized search.

    # Greedy: try to add moduli one by one, choosing a residue that works with all already-selected.
    # For a new modulus m with existing set S, we need r such that for all (r_i, m_i) in S:
    #   gcd(m, m_i) ∤ (r - r_i)
    # i.e., r mod gcd(m, m_i) != r_i mod gcd(m, m_i)

    # Let's try a randomized greedy with multiple restarts.

    def is_valid_assignment(selected):
        """Check if a list of [residue, modulus] pairs are all pairwise disjoint."""
        n = len(selected)
        for i in range(n):
            r_i, m_i = selected[i]
            for j in range(i + 1, n):
                r_j, m_j = selected[j]
                g = gcd(m_i, m_j)
                if (r_i - r_j) % g == 0:
                    return False
        return True

    def try_greedy(moduli_order, rng):
        """Greedy: add moduli in given order, pick a valid residue if possible."""
        selected = []
        for m in moduli_order:
            # Try all residues for this modulus
            found = False
            for r in range(m):
                # Check against all existing
                ok = True
                for (r_i, m_i) in selected:
                    g = gcd(m, m_i)
                    if (r - r_i) % g == 0:
                        ok = False
                        break
                if ok:
                    selected.append([r, m])
                    found = True
                    break
            # If no residue found, skip this modulus
        return selected

    def local_search(selected, rng, max_iters=20000):
        """Try to improve by swapping residues or adding/removing moduli."""
        current = [list(x) for x in selected]
        current_size = len(current)
        best = [list(x) for x in current]
        best_size = current_size
        moduli_in = set(m for _, m in current)
        moduli_out = [m for m in moduli if m not in moduli_in]

        for it in range(max_iters):
            if time.time() - start_time > DEADLINE - 2:
                break

            # Randomly pick an operation
            op = rng.random()
            if op < 0.5 and moduli_out:
                # Try to add a random modulus from moduli_out
                m = rng.choice(moduli_out)
                # Try random residues
                tried = set()
                while len(tried) < min(m, 50):
                    r = rng.randrange(m)
                    if r in tried:
                        continue
                    tried.add(r)
                    ok = True
                    for (r_i, m_i) in current:
                        g = gcd(m, m_i)
                        if (r - r_i) % g == 0:
                            ok = False
                            break
                    if ok:
                        new_sel = current + [[r, m]]
                        if len(new_sel) > best_size:
                            best = [list(x) for x in new_sel]
                            best_size = len(new_sel)
                        current = new_sel
                        moduli_in.add(m)
                        moduli_out.remove(m)
                        break
            elif op < 0.8:
                # Try to change a residue of an existing modulus
                if not current:
                    continue
                idx = rng.randrange(len(current))
                r_old, m = current[idx]
                # Try a few random new residues
                for _ in range(20):
                    r_new = rng.randrange(m)
                    if r_new == r_old:
                        continue
                    # Check validity with all others
                    ok = True
                    for j, (r_i, m_i) in enumerate(current):
                        if j == idx:
                            continue
                        g = gcd(m, m_i)
                        if (r_new - r_i) % g == 0:
                            ok = False
                            break
                    if ok:
                        current[idx][0] = r_new
                        break
            else:
                # Try to remove a modulus and re-add a different one
                if not current:
                    continue
                idx = rng.randrange(len(current))
                removed = current.pop(idx)
                moduli_in.discard(removed[1])
                moduli_out.append(removed[1])
                # Try to add a random modulus
                if moduli_out:
                    m = rng.choice(moduli_out)
                    for _ in range(50):
                        r = rng.randrange(m)
                        ok = True
                        for (r_i, m_i) in current:
                            g = gcd(m, m_i)
                            if (r - r_i) % g == 0:
                                ok = False
                                break
                        if ok:
                            current.append([r, m])
                            moduli_in.add(m)
                            moduli_out.remove(m)
                            if len(current) > best_size:
                                best = [list(x) for x in current]
                                best_size = len(current)
                            break

        return best

    best_result = []
    best_size = 0

    # Multiple randomized restarts with different orderings
    rng = random.Random(42)

    for restart in range(50):
        if time.time() - start_time > DEADLINE - 5:
            break

        # Shuffle moduli, maybe with some bias
        order = list(moduli)
        rng.shuffle(order)

        # Greedy
        sel = try_greedy(order, rng)
        if len(sel) > best_size:
            best_size = len(sel)
            best_result = [list(x) for x in sel]

        # Local search
        improved = local_search(sel, rng, max_iters=5000)
        if len(improved) > best_size:
            best_size = len(improved)
            best_result = [list(x) for x in improved]

    # Final polish: try to add any remaining moduli
    if time.time() - start_time < DEADLINE - 3:
        moduli_in = set(m for _, m in best_result)
        moduli_out = [m for m in moduli if m not in moduli_in]
        rng2 = random.Random(123)
        for m in moduli_out:
            if time.time() - start_time > DEADLINE - 3:
                break
            for r in range(m):
                ok = True
                for (r_i, m_i) in best_result:
                    g = gcd(m, m_i)
                    if (r - r_i) % g == 0:
                        ok = False
                        break
                if ok:
                    best_result.append([r, m])
                    break

    # Ensure canonical form and distinct moduli
    # The result should already be valid, but let's verify
    if not best_result:
        # Fallback: just return the seed
        return [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]

    return best_result
