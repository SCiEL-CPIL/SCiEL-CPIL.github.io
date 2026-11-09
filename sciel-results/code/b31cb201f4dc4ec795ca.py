"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute 4-AP tuples for fast violation checking
    # A 4-AP is (a, a+d, a+2d, a+3d) with a>=1, a+3d<=N
    four_ap_tuples = []
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            four_ap_tuples.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Precompute 3-AP tuples for counting
    # A 3-AP is (a, a+d, a+2d) with a>=1, a+2d<=N
    three_ap_tuples = []
    for d in range(1, (N - 1) // 2 + 1):
        max_a = N - 2 * d
        for a in range(1, max_a + 1):
            three_ap_tuples.append((a, a + d, a + 2 * d))
    
    def count_3aps(selected_set):
        """Count distinct 3-APs in selected_set."""
        count = 0
        for (a, b, c) in three_ap_tuples:
            if a in selected_set and b in selected_set and c in selected_set:
                count += 1
        return count
    
    def has_4ap(selected_set):
        """Check if selected_set contains any 4-AP."""
        for (a, b, c, d) in four_ap_tuples:
            if a in selected_set and b in selected_set and c in selected_set and d in selected_set:
                return True
        return False
    
    def can_add_without_4ap(selected_set, x):
        """Check if adding x to selected_set would create a 4-AP."""
        new_set = selected_set | {x}
        return not has_4ap(new_set)
    
    # Greedy construction: start with empty set, add elements that maximize 3-APs
    # while maintaining 4-AP-free property
    selected = set()
    elements = list(range(1, N + 1))
    
    while elements and time.time() < deadline - 5.0:
        best_elem = None
        best_score = -1
        
        for x in elements:
            if not can_add_without_4ap(selected, x):
                continue
            
            # Count 3-APs that would be created by adding x
            new_set = selected | {x}
            score = count_3aps(new_set)
            
            if score > best_score:
                best_score = score
                best_elem = x
        
        if best_elem is None:
            break
        
        selected.add(best_elem)
        elements.remove(best_elem)
    
    # Simulated annealing refinement
    current_solution = set(selected)
    current_score = count_3aps(current_solution)
    best_solution = set(current_solution)
    best_score = current_score
    
    if time.time() < deadline - 2.0:
        T = 50.0
        T_min = 0.1
        cooling_rate = 0.999
        
        while time.time() < deadline - 2.0 and T > T_min:
            # Random move: remove an element, or swap
            if random.random() < 0.5 and len(current_solution) > 0:
                # Remove a random element
                elem_to_remove = random.choice(list(current_solution))
                new_solution = current_solution - {elem_to_remove}
                new_score = count_3aps(new_solution)
                
                if new_score >= current_score or random.random() < math.exp((new_score - current_score) / T):
                    current_solution = new_solution
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_solution = set(current_solution)
            else:
                # Try to add a random element not in current_solution
                available = [x for x in range(1, N + 1) if x not in current_solution]
                if available:
                    elem_to_add = random.choice(available)
                    new_solution = current_solution | {elem_to_add}
                    if not has_4ap(new_solution):
                        new_score = count_3aps(new_solution)
                        if new_score >= current_score or random.random() < math.exp((new_score - current_score) / T):
                            current_solution = new_solution
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_solution = set(current_solution)
            
            T *= cooling_rate
    
    return sorted(best_solution)
# EVOLVE-BLOCK-END
