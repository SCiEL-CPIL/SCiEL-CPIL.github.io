import time
import itertools
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    
    # Precompute pairs for each triple
    triple_pairs = {t: set(itertools.combinations(t, 2)) for t in all_triples}

    # Helper to check if a set of points is independent (contains no triple)
    # Uses a fast bit-mask check against the list of triples
    # For speed, we precompute the mask for each triple
    triple_masks = [sum(1 << p for p in t) for t in all_triples]
    
    def is_independent(mask, triples_to_check=None):
        """
        Check if the set of points represented by 'mask' contains no triple from 'triples_to_check'.
        If triples_to_check is None, it checks against all_triples (not used in this context, 
        we always pass the current system's triples).
        """
        if triples_to_check is None:
            triples_to_check = all_triples
        for t_mask in triples_to_check:
            if (mask & t_mask) == t_mask:
                return False
        return True

    def get_alpha(triples):
        """
        Compute the independence number of the hypergraph defined by 'triples'.
        Uses a branch and bound / backtracking approach to find the largest independent set.
        Since n=16, we can iterate over subsets by size from 16 down to 1.
        """
        if not triples:
            return 16
            
        t_masks = [sum(1 << p for p in t) for t in triples]
        
        # Check sizes from 16 down to 1
        # For each size k, iterate through all combinations of points of size k
        # and check if any is independent.
        # Optimization: If we find one, return k.
        
        # Precompute all masks for each size? No, just generate on the fly.
        # 16 choose 8 is 12870, which is small.
        
        for k in range(16, 0, -1):
            for comb in itertools.combinations(range(16), k):
                mask = 0
                for p in comb:
                    mask |= (1 << p)
                if is_independent(mask, t_masks):
                    return k
        return 0

    # Strategy:
    # 1. Generate random greedy packings to initialize.
    # 2. Local search with simulated annealing acceptance.

    best_alpha = 16
    best_system = []

    # Initialize with random greedy packings
    for _ in range(20):
        if time.time() - start_time > deadline:
            break
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        current = []
        used_pairs = set()
        for t in shuffled:
            if all(p not in used_pairs for p in triple_pairs[t]):
                current.append(t)
                for p in triple_pairs[t]:
                    used_pairs.add(p)

        alpha = get_alpha(current)
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = current[:]

        if best_alpha <= 3:
            break

    current_system = best_system[:]
    current_alpha = best_alpha

    # Local search with simulated annealing
    def can_add_fast(triple, used_pairs):
        return all(p not in used_pairs for p in triple_pairs[triple])

    iteration = 0
    while time.time() - start_time < deadline:
        iteration += 1

        # Temperature decay for simulated annealing
        accept_prob = 1.0 / (1.0 + iteration / 500.0)

        move_type = random.choice(['add', 'swap', 'remove'])

        if move_type == 'add':
            # Find valid triples to add
            used_pairs = set()
            for t in current_system:
                used_pairs.update(triple_pairs[t])

            # Find candidates: triples that don't use any used pair
            candidates = [t for t in all_triples if all(p not in used_pairs for p in triple_pairs[t])]

            if candidates:
                new_t = random.choice(candidates)
                new_system = current_system + [new_t]
                new_alpha = get_alpha(new_system)

                if new_alpha < current_alpha:
                    current_system = new_system
                    current_alpha = new_alpha
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_system = current_system[:]
                elif new_alpha == current_alpha:
                    if random.random() < accept_prob:
                        current_system = new_system

        elif move_type == 'swap':
            if not current_system:
                continue

            idx = random.randint(0, len(current_system) - 1)
            t_remove = current_system[idx]

            # Build used_pairs excluding t_remove
            used_pairs = set()
            for i, t in enumerate(current_system):
                if i != idx:
                    used_pairs.update(triple_pairs[t])

            # Candidates: triples that don't use any used pair (and are not t_remove)
            candidates = [t for t in all_triples if all(p not in used_pairs for p in triple_pairs[t])]

            if candidates:
                new_t = random.choice(candidates)
                if new_t == t_remove:
                    continue

                new_system = current_system[:]
                new_system[idx] = new_t
                new_alpha = get_alpha(new_system)

                if new_alpha < current_alpha:
                    current_system = new_system
                    current_alpha = new_alpha
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_system = current_system[:]
                elif new_alpha == current_alpha:
                    if random.random() < accept_prob:
                        current_system = new_system
                        
        elif move_type == 'remove':
            # Sometimes removing a triple might allow adding two others later?
            # Or just to explore the space.
            if len(current_system) > 1:
                idx = random.randint(0, len(current_system) - 1)
                new_system = current_system[:]
                new_system.pop(idx)
                new_alpha = get_alpha(new_system)
                
                # We generally don't want to remove unless it helps, but in SA we can accept
                # worse solutions to escape local minima. However, removing a triple
                # usually *increases* or keeps the same alpha (makes it easier to be independent).
                # So new_alpha >= current_alpha is likely. 
                # We only accept if it leads to a better path later? 
                # Actually, removing a triple makes the constraint looser, so alpha cannot decrease.
                # So this move is never an improvement in alpha. 
                # It might be useful if we are stuck and want to rearrange? 
                # But 'swap' covers rearrangement. 'remove' is likely useless for minimization.
                # Let's skip 'remove' as a primary move type for minimization, 
                # or only use it if we allow temporary worsening? 
                # Since alpha is monotonic with respect to adding constraints (triples),
                # removing a triple relaxes constraints, so alpha stays same or increases.
                # Thus, it never improves the score. We can skip this move type to save time.
                pass

        # Occasional random restart to escape deep local minima
        if iteration % 2000 == 0 and best_alpha > 3:
            # Try a fresh greedy initialization
            shuffled = all_triples[:]
            random.shuffle(shuffled)
            restart_sys = []
            used_pairs = set()
            for t in shuffled:
                if all(p not in used_pairs for p in triple_pairs[t]):
                    restart_sys.append(t)
                    for p in triple_pairs[t]:
                        used_pairs.add(p)
            r_alpha = get_alpha(restart_sys)
            if r_alpha < current_alpha:
                current_system = restart_sys
                current_alpha = r_alpha
                if r_alpha < best_alpha:
                    best_alpha = r_alpha
                    best_system = restart_sys[:]

        if best_alpha <= 3:
            break

    return [list(t) for t in best_system]

if __name__ == "__main__":
    res = construct()
    print(res)
