import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin (evaluator kills at 90s)
    
    MAX_MOD = 128
    moduli = list(range(2, MAX_MOD + 1))
    
    def is_disjoint(r1, m1, r2, m2):
        d = gcd(m1, m2)
        return (r1 - r2) % d != 0
    
    def is_valid_with_all(r, m, selected):
        for [r_s, m_s] in selected:
            if not is_disjoint(r, m, r_s, m_s):
                return False
        return True
    
    def try_greedy(order, seed=0):
        """Greedy construction: for each modulus in order, pick a valid residue."""
        rng = random.Random(seed)
        selected = []
        for m in order:
            candidates = list(range(m))
            rng.shuffle(candidates)
            # Try a few random residues, then fall back to systematic scan
            for r in candidates:
                if is_valid_with_all(r, m, selected):
                    selected.append([r, m])
                    break
            else:
                # Systematic scan as fallback
                for r in range(m):
                    if is_valid_with_all(r, m, selected):
                        selected.append([r, m])
                        break
        return selected
    
    def local_search(selected, seed=0):
        """Try to add unselected moduli by swapping residues or reordering."""
        rng = random.Random(seed)
        selected_set = set(m for _, m in selected)
        unselected = [m for m in moduli if m not in selected_set]
        
        # Try to add one unselected modulus at a time
        improved = True
        while improved:
            improved = False
            for m_add in list(unselected):
                # Find a residue for m_add that's disjoint from all selected
                for r in range(m_add):
                    if is_valid_with_all(r, m_add, selected):
                        selected.append([r, m_add])
                        selected_set.add(m_add)
                        unselected.remove(m_add)
                        improved = True
                        break
                if improved:
                    break
        
        # Try swapping: remove one selected, try to add two unselected
        for _ in range(50):
            if not unselected:
                break
            # Pick a random selected to potentially remove
            idx = rng.randrange(len(selected))
            r_old, m_old = selected[idx]
            selected.remove([r_old, m_old])
            selected_set.discard(m_old)
            
            # Try to add two unselected moduli
            added = 0
            for m_add in list(unselected):
                for r in range(m_add):
                    if is_valid_with_all(r, m_add, selected):
                        selected.append([r, m_add])
                        selected_set.add(m_add)
                        unselected.remove(m_add)
                        added += 1
                        break
                if added >= 2:
                    break
            
            # If we didn't add at least 2, revert
            if added < 2:
                # Revert: remove what we added, put back old
                for m_add in [m for _, m in selected if m not in selected_set or True]:
                    pass
                # This is getting complicated; let's simplify
                # Actually, let's just try a simpler swap strategy
        
        return selected
    
    def local_search_v2(selected, seed=0):
        """Simpler local search: try to add unselected moduli."""
        rng = random.Random(seed)
        selected_set = set(m for _, m in selected)
        unselected = [m for m in moduli if m not in selected_set]
        
        # Repeatedly try to add any unselected modulus
        while unselected:
            progress = False
            # Shuffle unselected for variety
            rng.shuffle(unselected)
            for m_add in unselected[:]:
                # Try residues in random order
                residues = list(range(m_add))
                rng.shuffle(residues)
                for r in residues:
                    if is_valid_with_all(r, m_add, selected):
                        selected.append([r, m_add])
                        selected_set.add(m_add)
                        unselected.remove(m_add)
                        progress = True
                        break
            if not progress:
                break
        
        # Try to swap: remove one, add one or more
        for _ in range(100):
            if not unselected:
                break
            idx = rng.randrange(len(selected))
            old_pair = selected[idx]
            m_old = old_pair[1]
            selected.remove(old_pair)
            selected_set.discard(m_old)
            
            added_count = 0
            for m_add in list(unselected):
                residues = list(range(m_add))
                rng.shuffle(residues)
                for r in residues:
                    if is_valid_with_all(r, m_add, selected):
                        selected.append([r, m_add])
                        selected_set.add(m_add)
                        unselected.remove(m_add)
                        added_count += 1
                        break
            
            if added_count < 1:
                # Revert
                selected.append(old_pair)
                selected_set.add(m_old)
        
        return selected
    
    # Initial deterministic attempts with different orderings
    orders = [
        list(reversed(moduli)),  # descending
        list(moduli),  # ascending
        sorted(moduli, key=lambda x: x.bit_count(), reverse=True),
        sorted(moduli, key=lambda x: x.bit_count()),
        sorted(moduli, key=lambda x: sum(1 for i in range(2, x) if x % i == 0)),  # by number of divisors
        sorted(moduli, key=lambda x: sum(1 for i in range(2, x) if x % i == 0), reverse=True),
    ]
    
    best_solution = []
    best_size = 0
    
    # Try deterministic orders
    for i, order in enumerate(orders):
        if time.time() > deadline:
            break
        sol = try_greedy(order, seed=i)
        sol = local_search_v2(sol, seed=i)
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = sol[:]
    
    # Random restarts with greedy + local search
    seed_counter = 100
    while time.time() < deadline - 5:
        order = list(moduli)
        random.shuffle(order)
        sol = try_greedy(order, seed=seed_counter)
        sol = local_search_v2(sol, seed=seed_counter)
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = sol[:]
            # If we found a great solution, we can be more aggressive
        seed_counter += 1
    
    # Final polish: try to add any remaining moduli to the best solution
    if best_solution:
        best_set = set(m for _, m in best_solution)
        remaining = [m for m in moduli if m not in best_set]
        for m_add in remaining:
            if time.time() > deadline:
                break
            for r in range(m_add):
                if is_valid_with_all(r, m_add, best_solution):
                    best_solution.append([r, m_add])
                    best_set.add(m_add)
                    break
    
    return best_solution if best_solution else [[1, 2]]
