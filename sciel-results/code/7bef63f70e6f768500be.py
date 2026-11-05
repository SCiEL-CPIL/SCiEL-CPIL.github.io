import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    def get_triples(n):
        return list(combinations(range(n), 3))
    
    def precompute(n):
        triples = get_triples(n)
        triple_index = {t: i for i, t in enumerate(triples)}
        
        four_sets = list(combinations(range(n), 4))
        five_sets = list(combinations(range(n), 5))
        
        # Map each triple to the 4-sets and 5-sets it belongs to
        triple_to_4sets = [[] for _ in range(len(triples))]
        triple_to_5sets = [[] for _ in range(len(triples))]
        
        for i, s4 in enumerate(four_sets):
            for t in combinations(s4, 3):
                triple_to_4sets[triple_index[t]].append(i)
                
        for i, s5 in enumerate(five_sets):
            for t in combinations(s5, 3):
                triple_to_5sets[triple_index[t]].append(i)
                
        return triples, triple_index, four_sets, five_sets, triple_to_4sets, triple_to_5sets

    def initial_coloring_structured(n):
        """
        Generate structured initial colorings to avoid random traps.
        Use algebraic constructions that are known to work well for hypergraph Ramsey problems.
        """
        colors = []
        for a, b, c in combinations(range(n), 3):
            # Use a combination of modular arithmetic and threshold
            # Mode: Use quadratic form mod 2 for good mixing
            val = (a * b + b * c + c * a + a * b * c) % 2
            # Add some structure based on position
            if (a + b + c) % 3 == 0:
                val ^= 1
            colors.append(val)
        return colors

    def initial_coloring_modular(n):
        """
        Another structured initialization using different modular arithmetic.
        """
        colors = []
        for a, b, c in combinations(range(n), 3):
            # Use sum mod 3, mapped to 0/1
            s = (a + b + c) % 3
            if s == 0:
                col = 0
            elif s == 1:
                col = 1
            else:
                # Use product for the remaining case
                col = (a * b * c) % 2
            colors.append(col)
        return colors

    def solve_for_n(n, time_limit):
        triples, triple_index, four_sets, five_sets, triple_to_4sets, triple_to_5sets = precompute(n)
        num_triples = len(triples)
        num_4sets = len(four_sets)
        num_5sets = len(five_sets)
        
        # Try multiple initializations
        best_result = None
        best_cost = float('inf')
        
        # Prioritize structured initializations
        modes = [0, 1, 0, 1, 2, 2]  # 0: structured, 1: modular, 2: random
        random.shuffle(modes)
        
        for mode in modes:
            if time.time() > time_limit:
                break
                
            if mode == 0:
                colors = initial_coloring_structured(n)
            elif mode == 1:
                colors = initial_color_coloring_modular(n)
            else:
                colors = [random.randint(0, 1) for _ in range(num_triples)]
            
            # State: red_count_4, blue_count_5
            red_count_4 = [0] * num_4sets
            blue_count_5 = [0] * num_5sets
            
            # Initialize counts
            for t_idx in range(num_triples):
                is_red = (colors[t_idx] == 0)
                for s4_idx in triple_to_4sets[t_idx]:
                    if is_red:
                        red_count_4[s4_idx] += 1
                for s5_idx in triple_to_5sets[t_idx]:
                    if not is_red:
                        blue_count_5[s5_idx] += 1
            
            # Priority Queue of violations
            # (cost, type, index)
            # Type 0: Red K4 violation (red_count_4 == 4)
            # Type 1: Blue K5 violation (blue_count_5 == 10)
            import heapq
            pq = []
            
            for i in range(num_4sets):
                if red_count_4[i] == 4:
                    heapq.heappush(pq, (4, 0, i))
            
            for j in range(num_5sets):
                if blue_count_5[j] == 10:
                    heapq.heappush(pq, (10, 1, j))
            
            current_cost = len(pq)
            
            if current_cost == 0:
                return colors
            
            # Local Search Loop
            steps = 0
            max_steps = 100000  # Hard cap to prevent infinite loops if stuck
            
            while time.time() < time_limit and steps < max_steps:
                if not pq:
                    break
                    
                # Get a violation
                popped = None
                while pq:
                    cost, typ, idx = heapq.heappop(pq)
                    # Check if still valid
                    if typ == 0:
                        if red_count_4[idx] == 4:
                            popped = (typ, idx)
                            break
                    else:
                        if blue_count_5[idx] == 10:
                            popped = (typ, idx)
                            break
                
                if popped is None:
                    break  # No more violations? Or all stale?
                    
                typ, idx = popped
                
                # Find a triple to flip within this set to reduce violation
                # Prioritize red K4 violations since they're rarer
                if typ == 0:
                    s4 = four_sets[idx]
                    candidates = []
                    for t in combinations(s4, 3):
                        t_idx = triple_index[t]
                        if colors[t_idx] == 0:  # Is Red
                            candidates.append(t_idx)
                    
                    if not candidates:
                        continue  # Should not happen if count was 4
                        
                    # Pick a candidate to flip
                    # Heuristic: Pick the one that causes least new damage
                    t_idx = random.choice(candidates)
                    
                else:
                    s5 = five_sets[idx]
                    candidates = []
                    for t in combinations(s5, 3):
                        t_idx = triple_index[t]
                        if colors[t_idx] == 1:  # Is Blue
                            candidates.append(t_idx)
                    
                    if not candidates:
                        continue
                        
                    t_idx = random.choice(candidates)
                
                # Perform Flip
                old_color = colors[t_idx]
                new_color = 1 - old_color
                
                # Update counts and PQ
                # We need to re-push affected sets if their status changes
                
                # Red count changes
                for s4_idx in triple_to_4sets[t_idx]:
                    old_c = red_count_4[s4_idx]
                    if old_color == 0:  # R->B
                        red_count_4[s4_idx] = old_c - 1
                        # If it was 4, it's now 3 (fixed). If it was 3, now 2.
                    else:  # B->R
                        red_count_4[s4_idx] = old_c + 1
                        # If old_c was 3, new is 4. New violation!
                        if old_c == 3:
                            heapq.heappush(pq, (4, 0, s4_idx))
                            current_cost += 1
                        
                # Blue count changes
                for s5_idx in triple_to_5sets[t_idx]:
                    old_c = blue_count_5[s5_idx]
                    if old_color == 1:  # B->R
                        blue_count_5[s5_idx] = old_c - 1
                        # If old_c was 10, new is 9. Fixed.
                    else:  # R->B
                        blue_count_5[s5_idx] = old_c + 1
                        # If old_c was 9, new is 10. New violation!
                        if old_c == 9:
                            heapq.heappush(pq, (10, 1, s5_idx))
                            current_cost += 1
                        
                # Apply flip
                colors[t_idx] = new_color
                
                steps += 1
                
                if current_cost == 0:
                    return colors
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_result = colors.copy()
                
            if best_cost == 0:
                return best_result
                
        if best_result is not None and best_cost == 0:
            return best_result
        return None

    # Try n from 34 downwards
    # Allocate time based on difficulty
    time_alloc = {
        34: 70.0,
        33: 15.0,
        32: 8.0,
        31: 4.0,
        30: 2.0,
        29: 1.0,
        28: 0.5
    }
    
    for n in range(34, 8, -1):
        if time.time() > deadline - 3.0:
            break
            
        t_limit = time_alloc.get(n, 1.0)
        t_start = time.time()
        
        # Cap t_limit by remaining time
        remaining = deadline - 3.0 - t_start
        if remaining <= 0:
            break
        t_limit = min(t_limit, remaining)
        
        res = solve_for_n(n, t_start + t_limit)
        if res is not None:
            # Verify validity before returning
            if verify(n, res):
                return {'n': n, 'colors': res}
            else:
                # If invalid, continue to next n
                pass

    # Fallback
    n = 4
    colors = [1, 0, 0, 0]
    return {'n': n, 'colors': colors}

def verify(n, colors):
    """
    Verify if a coloring is valid.
    colors: list of 0/1 for all triples in lex order.
    """
    triples = list(combinations(range(n), 3))
    triple_index = {t: i for i, t in enumerate(triples)}
    
    # Check Red K4
    for s4 in combinations(range(n), 4):
        is_red = True
        for t in combinations(s4, 3):
            if colors[triple_index[t]] != 0:
                is_red = False
                break
        if is_red:
            return False
            
    # Check Blue K5
    for s5 in combinations(range(n), 5):
        is_blue = True
        for t in combinations(s5, 3):
            if colors[triple_index[t]] != 1:
                is_blue = False
                break
        if is_blue:
            return False
            
    return True

if __name__ == "__main__":
    res = construct()
    print(res['n'])
