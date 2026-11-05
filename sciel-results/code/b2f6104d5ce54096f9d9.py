"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    all_moduli = list(range(20, 201))
    n_moduli = len(all_moduli)

    # Precompute gcd table for all pairs of moduli
    gcd_table = {}
    for i, m1 in enumerate(all_moduli):
        for m2 in all_moduli[i+1:]:
            gcd_table[(m1, m2)] = math.gcd(m1, m2)
            gcd_table[(m2, m1)] = math.gcd(m1, m2)
        gcd_table[(m1, m1)] = m1

    def get_gcd(m1, m2):
        if m1 == m2:
            return m1
        key = (m1, m2) if m1 < m2 else (m2, m1)
        return gcd_table.get(key, math.gcd(m1, m2))

    def is_compatible(m_new, r_new, selected):
        """Check if (m_new, r_new) is disjoint from all in selected."""
        for m_old, r_old in selected:
            g = get_gcd(m_new, m_old)
            if (r_new - r_old) % g == 0:
                return False
        return True

    def get_valid_residues(m, selected):
        """Get all valid residues for modulus m given selected classes."""
        valid = []
        for r in range(m):
            if is_compatible(m, r, selected):
                valid.append(r)
        return valid

    def score(classes):
        """Compute the score (sum of 1/m)."""
        return sum(1.0 / m for m, r in classes)

    def greedy_build(moduli_order, randomize=True):
        """Build a solution by processing moduli in given order."""
        selected = []
        for m in moduli_order:
            valid = get_valid_residues(m, selected)
            if valid:
                if randomize and len(valid) > 1:
                    r = random.choice(valid)
                else:
                    r = valid[0]
                selected.append([m, r])
        return selected

    def simulated_annealing(initial, max_iter=5000):
        """Simulated annealing to optimize the solution."""
        current = [c[:] for c in initial]
        current_score = score(current)
        best = [c[:] for c in current]
        best_score = current_score

        T = 0.5  # Initial temperature
        cooling = 0.9995  # Cooling rate

        used_moduli = set(m for m, r in current)
        unused_moduli = [m for m in all_moduli if m not in used_moduli]

        for it in range(max_iter):
            if time.time() > deadline - 5:
                break

            move_type = random.random()
            new_solution = [c[:] for c in current]
            new_score = current_score

            if move_type < 0.3:
                # Change residue of an existing class
                if not current:
                    continue
                idx = random.randint(0, len(current) - 1)
                m, r = current[idx]
                valid = get_valid_residues(m, [c for i, c in enumerate(current) if i != idx])
                if valid:
                    new_r = random.choice(valid)
                    if new_r != r:
                        new_solution[idx] = [m, new_r]
                        new_score = score(new_solution)

            elif move_type < 0.5:
                # Add a new modulus
                if unused_moduli:
                    m_new = random.choice(unused_moduli)
                    valid = get_valid_residues(m_new, current)
                    if valid:
                        r_new = random.choice(valid)
                        new_solution.append([m_new, r_new])
                        new_score = score(new_solution)
                        used_moduli.add(m_new)
                        unused_moduli.remove(m_new)

            elif move_type < 0.7:
                # Remove a modulus
                if len(current) > 1:
                    idx = random.randint(0, len(current) - 1)
                    m_removed = current[idx][0]
                    new_solution.pop(idx)
                    new_score = score(new_solution)
                    used_moduli.discard(m_removed)
                    unused_moduli.append(m_removed)

            else:
                # Swap two moduli
                if len(current) >= 2:
                    i, j = random.sample(range(len(current)), 2)
                    m_i, r_i = current[i]
                    m_j, r_j = current[j]
                    # Try swapping residues
                    valid_i = get_valid_residues(m_i, [c for k, c in enumerate(current) if k != i])
                    valid_j = get_valid_residues(m_j, [c for k, c in enumerate(current) if k != j])
                    if r_j in valid_i and r_i in valid_j:
                        new_solution[i] = [m_i, r_j]
                        new_solution[j] = [m_j, r_i]
                        new_score = score(new_solution)

            delta = new_score - current_score
            if delta >= 0 or random.random() < math.exp(delta / T):
                current = new_solution
                current_score = new_score
                if current_score > best_score:
                    best = [c[:] for c in current]
                    best_score = current_score

            T *= cooling

        return best, best_score

    # Best solution found
    best_score = 0.0
    best_classes = []

    # First, try a deterministic greedy: sort by m ascending
    moduli_sorted = sorted(all_moduli)
    classes = greedy_build(moduli_sorted, randomize=False)
    s = score(classes)
    if s > best_score:
        best_score = s
        best_classes = classes

    # Try multiple randomized restarts with SA
    random.seed(42)
    num_restarts = 0
    while time.time() < deadline - 20:
        # Random shuffle of moduli
        shuffled = all_moduli[:]
        random.shuffle(shuffled)
        classes = greedy_build(shuffled, randomize=True)
        s = score(classes)
        if s > best_score:
            best_score = s
            best_classes = classes

        # Run SA on this initial solution
        improved, improved_score = simulated_annealing(classes, max_iter=3000)
        if improved_score > best_score:
            best_score = improved_score
            best_classes = improved
        num_restarts += 1

    # Also try a heuristic: prioritize primes and small numbers
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    primes_in_range = [m for m in all_moduli if is_prime(m)]
    composites_in_range = [m for m in all_moduli if not is_prime(m)]

    # Try: primes first (sorted by m), then composites
    order1 = sorted(primes_in_range) + sorted(composites_in_range)
    classes = greedy_build(order1, randomize=False)
    s = score(classes)
    if s > best_score:
        best_score = s
        best_classes = classes
    improved, improved_score = simulated_annealing(classes, max_iter=5000)
    if improved_score > best_score:
        best_score = improved_score
        best_classes = improved

    # Try: all sorted by m, but with some randomization in residue choice
    for _ in range(20):
        if time.time() >= deadline - 10:
            break
        classes = greedy_build(moduli_sorted, randomize=True)
        s = score(classes)
        if s > best_score:
            best_score = s
            best_classes = classes
        improved, improved_score = simulated_annealing(classes, max_iter=2000)
        if improved_score > best_score:
            best_score = improved_score
            best_classes = improved

    # Final local search: try adding any missing modulus
    if time.time() < deadline - 2:
        used_moduli = set(m for m, r in best_classes)
        for m_new in all_moduli:
            if m_new not in used_moduli:
                valid = get_valid_residues(m_new, best_classes)
                if valid:
                    new_classes = best_classes + [[m_new, valid[0]]]
                    s = score(new_classes)
                    if s > best_score:
                        best_score = s
                        best_classes = new_classes

    # Verify the best solution is valid
    if best_classes:
        for i in range(len(best_classes)):
            for j in range(i + 1, len(best_classes)):
                m1, r1 = best_classes[i]
                m2, r2 = best_classes[j]
                g = get_gcd(m1, m2)
                if (r1 - r2) % g == 0:
                    # Invalid - should not happen, but handle gracefully
                    return [[20, 0]]

    return best_classes if best_classes else [[20, 0]]
# EVOLVE-BLOCK-END
