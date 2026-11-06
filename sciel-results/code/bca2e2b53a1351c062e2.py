"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    """
    Maximizes sum(1/m) for pairwise disjoint residue classes with distinct moduli in [20, 200].
    
    Strategy:
    1. Precompute gcds and conflict logic.
    2. Greedy construction with multiple orderings.
    3. Local search with rigorous state management to ensure distinct moduli.
    4. Simulated annealing for global exploration.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    def time_left():
        return deadline - time.time()
    
    # Precompute gcds for all pairs in [20, 200]
    moduli_range = range(20, 201)
    gcd_table = {}
    for m1 in moduli_range:
        for m2 in moduli_range:
            if m1 not in gcd_table:
                gcd_table[m1] = {}
            gcd_table[m1][m2] = gcd(m1, m2)

    def is_valid(m, r, existing):
        """
        Check if (m, r) is disjoint from all existing (m_i, r_i) 
        AND m is not already in existing (distinct moduli constraint).
        """
        for m_i, r_i in existing:
            if m_i == m:
                return False
            g = gcd_table[m][m_i]
            if r % g == r_i % g:
                return False
        return True
    
    def find_valid_residue(m, existing, randomize=False):
        """Find a residue r in [0, m) such that (m, r) is disjoint from all existing."""
        residues = list(range(m))
        if randomize:
            random.shuffle(residues)
        for r in residues:
            if is_valid(m, r, existing):
                return r
        return None
    
    def calc_score(existing):
        """Calculate the score as sum of 1/m."""
        return sum(1.0 / m for m, _ in existing)
    
    def update_state(solution):
        """Recompute selected and missing moduli sets."""
        selected_moduli = set(m for m, _ in solution)
        missing_moduli = [m for m in range(20, 201) if m not in selected_moduli]
        return selected_moduli, missing_moduli

    def greedy_construct(order, use_random_residues=False):
        """Greedy construction with given modulus order."""
        solution = []
        for m in order:
            if time_left() < 5:
                break
            r = find_valid_residue(m, solution, randomize=use_random_residues)
            if r is not None:
                solution.append((m, r))
        return solution
    
    def local_search(solution, max_iterations=200):
        """Local search with add, swap, and reassign moves."""
        best_solution = list(solution)
        best_score = calc_score(best_solution)
        selected_moduli, missing_moduli = update_state(best_solution)
        
        for iteration in range(max_iterations):
            if time_left() < 10:
                break
            
            improved = False
            
            # 1. Try adding each missing modulus
            for m in missing_moduli:
                if time_left() < 10:
                    break
                r = find_valid_residue(m, best_solution)
                if r is not None:
                    new_sol = best_solution + [(m, r)]
                    new_score = calc_score(new_sol)
                    if new_score > best_score + 1e-12:
                        best_solution = new_sol
                        best_score = new_score
                        improved = True
                        selected_moduli, missing_moduli = update_state(best_solution)
            
            if not improved:
                # 2. Try swaps: remove one, add another
                # Sort by value to prioritize removing low-value items
                sorted_sol = sorted(best_solution, key=lambda x: 1.0/x[0])
                
                for m_remove, r_remove in sorted_sol:
                    if time_left() < 10:
                        break
                    
                    without_remove = [x for x in best_solution if x[0] != m_remove]
                    
                    for m_add in missing_moduli:
                        if 1.0/m_add <= 1.0/m_remove:
                            continue
                        
                        r_add = find_valid_residue(m_add, without_remove)
                        if r_add is not None:
                            new_sol = without_remove + [(m_add, r_add)]
                            new_score = calc_score(new_sol)
                            if new_score > best_score + 1e-12:
                                best_solution = new_sol
                                best_score = new_score
                                improved = True
                                selected_moduli, missing_moduli = update_state(best_solution)
                                break
            
            if not improved:
                # 3. Try reassigning residues
                # For each selected modulus, try to change its residue to allow adding others
                for idx, (m, r) in enumerate(best_solution):
                    if time_left() < 10:
                        break
                    
                    without_this = [x for j, x in enumerate(best_solution) if j != idx]
                    
                    # Try all residues for m
                    for new_r in range(m):
                        if new_r == r:
                            continue
                        if is_valid(m, new_r, without_this):
                            test_sol = without_this + [(m, new_r)]
                            
                            # Try to add missing moduli to test_sol
                            added_any = False
                            for m_add in missing_moduli:
                                r_add = find_valid_residue(m_add, test_sol)
                                if r_add is not None:
                                    test_sol = test_sol + [(m_add, r_add)]
                                    added_any = True
                            
                            if added_any:
                                new_score = calc_score(test_sol)
                                if new_score > best_score + 1e-12:
                                    best_solution = test_sol
                                    best_score = new_score
                                    improved = True
                                    selected_moduli, missing_moduli = update_state(best_solution)
                                    break
            
            if not improved:
                break
        
        return best_solution, best_score
    
    def simulated_annealing(solution, temperature=0.05, cooling_rate=0.99, max_iterations=1000):
        """Simulated annealing to escape local optima."""
        current_solution = list(solution)
        current_score = calc_score(current_solution)
        best_solution = list(current_solution)
        best_score = current_score
        selected_moduli, missing_moduli = update_state(current_solution)
        
        for iteration in range(max_iterations):
            if time_left() < 15:
                break
            
            temp = temperature * (cooling_rate ** iteration)
            
            # Choose a random move
            move_type = random.random()
            new_sol = None
            new_score = None
            
            if move_type < 0.3:
                # Add a missing modulus
                if missing_moduli:
                    m_add = random.choice(missing_moduli)
                    r_add = find_valid_residue(m_add, current_solution, randomize=True)
                    if r_add is not None:
                        new_sol = current_solution + [(m_add, r_add)]
                        new_score = calc_score(new_sol)
                        
            elif move_type < 0.7:
                # Swap: remove one, add another
                if missing_moduli and current_solution:
                    m_remove = random.choice(current_solution)[0]
                    m_add = random.choice(missing_moduli)
                    without_remove = [x for x in current_solution if x[0] != m_remove]
                    r_add = find_valid_residue(m_add, without_remove, randomize=True)
                    if r_add is not None:
                        new_sol = without_remove + [(m_add, r_add)]
                        new_score = calc_score(new_sol)
                        
            else:
                # Reassign a residue
                if current_solution:
                    idx = random.randint(0, len(current_solution) - 1)
                    m, r = current_solution[idx]
                    new_r = random.randint(0, m - 1)
                    if new_r != r:
                        without_this = [x for j, x in enumerate(current_solution) if j != idx]
                        if is_valid(m, new_r, without_this):
                            new_sol = without_this + [(m, new_r)]
                            new_score = calc_score(new_sol)
            
            if new_sol is None:
                continue
            
            # Accept or reject
            if new_score >= current_score:
                current_solution = new_sol
                current_score = new_score
                if new_score > best_score:
                    best_solution = list(new_sol)
                    best_score = new_score
                    selected_moduli, missing_moduli = update_state(best_solution)
            else:
                delta = new_score - current_score
                if delta > -1e-12:
                    current_solution = new_sol
                    current_score = new_score
                else:
                    # Simple acceptance probability
                    prob = 1.0 / (1.0 + abs(delta) / max(temp, 1e-10))
                    if random.random() < prob:
                        current_solution = new_sol
                        current_score = new_score
                        # Update state for best tracking? No, only update best if better.
                        # But we need to keep current_solution consistent for future moves.
                        # selected/missing for current_solution is not strictly needed for move gen 
                        # if we pass current_solution directly to find_valid_residue.
                        # However, we use missing_moduli for sampling. So update it.
                        selected_moduli, missing_moduli = update_state(current_solution)
        
        return best_solution, best_score
    
    # Step 1: Multiple greedy initializations
    best_solution = []
    best_score = 0.0
    
    base_moduli = list(range(20, 201))
    
    # Ordering 1: Ascending m (highest weight)
    order1 = sorted(base_moduli)
    sol1 = greedy_construct(order1)
    score1 = calc_score(sol1)
    if score1 > best_score:
        best_solution = sol1
        best_score = score1
    
    # Ordering 2: Descending m (lowest weight)
    order2 = sorted(base_moduli, reverse=True)
    sol2 = greedy_construct(order2)
    score2 = calc_score(sol2)
    if score2 > best_score:
        best_solution = sol2
        best_score = score2
        
    # Ordering 3: Random
    for _ in range(5):
        if time_left() < 10:
            break
        order3 = base_moduli.copy()
        random.shuffle(order3)
        sol3 = greedy_construct(order3, use_random_residues=True)
        score3 = calc_score(sol3)
        if score3 > best_score:
            best_solution = sol3
            best_score = score3
    
    # Step 2: Local search
    if time_left() > 10:
        best_solution, best_score = local_search(best_solution)
    
    # Step 3: Simulated annealing
    if time_left() > 20:
        sa_solution, sa_score = simulated_annealing(best_solution)
        if sa_score > best_score:
            best_solution = sa_solution
            best_score = sa_score
    
    # Step 4: Final local search
    if time_left() > 10:
        best_solution, best_score = local_search(best_solution)
    
    # Convert to required format [[m, r], ...]
    result = [[m, r] for m, r in best_solution]
    return result

# EVOLVE-BLOCK-END
