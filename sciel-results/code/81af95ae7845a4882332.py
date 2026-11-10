import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    def is_valid(selected, m, r):
        """Check if (m, r) is disjoint from all (m_j, r_j) in selected."""
        for m_j, r_j in selected:
            g = math.gcd(m, m_j)
            if r % g == r_j % g:
                return False
        return True
    
    def greedy_pass(random_seed=None):
        """Generates a solution using a greedy approach."""
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        selected = []
        for m in range(20, 201):
            if time.time() > deadline:
                break
            rs_to_check = list(range(m))
            if random_seed is not None:
                rng.shuffle(rs_to_check)
            for r in rs_to_check:
                if is_valid(selected, m, r):
                    selected.append((m, r))
                    break
        score = sum(1.0 / m for m, r in selected)
        return selected, score
    
    def local_search(selected, max_iterations=5000):
        """Improves a given solution using local search with multiple move types."""
        selected = [tuple(x) for x in selected]
        current_score = sum(1.0 / m for m, r in selected)
        best_score = current_score
        best_solution = list(selected)
        
        iteration = 0
        improved = True
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            if time.time() > deadline - 5:
                break
                
            # Try all possible moves:
            # 1. Change residue of an existing modulus
            # 2. Add a new modulus
            # 3. Remove a modulus and add a different one
            
            # Move type 1: Change residue
            for idx in range(len(selected)):
                m, r = selected[idx]
                others = selected[:idx] + selected[idx+1:]
                for new_r in range(m):
                    if new_r == r:
                        continue
                    if is_valid(others, m, new_r):
                        new_selected = others + [(m, new_r)]
                        new_score = current_score  # Same modulus, so score unchanged
                        # This move doesn't change score, so skip unless it enables other moves
                        # Actually, it might enable adding more moduli later
                        # For now, skip score-neutral moves in local search
                        pass
            
            # Move type 2: Add a new modulus
            m_set = {m for m, r in selected}
            for m_add in range(20, 201):
                if m_add in m_set:
                    continue
                for r in range(m_add):
                    if is_valid(selected, m_add, r):
                        new_selected = list(selected) + [(m_add, r)]
                        new_score = current_score + 1.0 / m_add
                        if new_score > best_score + 1e-12:
                            selected = new_selected
                            current_score = new_score
                            best_score = new_score
                            improved = True
                        break  # Only try first valid r for speed
            
            # Move type 3: Remove one and try to add multiple
            for idx in range(len(selected)):
                m_to_remove, r_to_remove = selected[idx]
                others = selected[:idx] + selected[idx+1:]
                new_selected = list(others)
                new_score = current_score - 1.0 / m_to_remove
                
                # Try to add any modulus not in new_selected
                m_set_new = {m for m, r in new_selected}
                for m_add in range(20, 201):
                    if m_add in m_set_new:
                        continue
                    for r in range(m_add):
                        if is_valid(new_selected, m_add, r):
                            new_selected.append((m_add, r))
                            new_score += 1.0 / m_add
                            m_set_new.add(m_add)
                            break
                
                if new_score > best_score + 1e-12:
                    selected = new_selected
                    current_score = new_score
                    best_score = new_score
                    improved = True
                    break
        
        return best_solution, best_score
    
    best_score = 0.0
    best_solution = []
    
    # 1. Run Deterministic Greedy
    sol, score = greedy_pass(random_seed=None)
    if score > best_score:
        best_score = score
        best_solution = sol
        
    # 2. Run Randomized Greedy Passes
    num_passes = 200
    for i in range(num_passes):
        if time.time() > deadline - 30:  # Reserve 30s for local search
            break
        seed = i
        sol, score = greedy_pass(random_seed=seed)
        if score > best_score:
            best_score = score
            best_solution = sol

    # 3. Apply Local Search to the best solution found
    if time.time() < deadline - 5:
        improved_sol, improved_score = local_search(best_solution)
        if improved_score > best_score:
            best_score = improved_score
            best_solution = improved_sol
    
    return [[m, r] for m, r in best_solution]
