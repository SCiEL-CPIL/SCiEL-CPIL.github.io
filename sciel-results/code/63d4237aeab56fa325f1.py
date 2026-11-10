"""Rewritten program with robust local search (Simulated Annealing) for edge selection
and a diverse set of value candidates."""

# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    maximum = 10000
    
    def _raw_value(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))

    def _evaluate_edges(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return len(sums), len(prods)

    # Generate diverse candidate value sets
    def generate_value_sets():
        sets = []
        
        # 1. Consecutive integers (AP with d=1)
        for start in range(1, 10):
            sets.append(list(range(start, start + 14)))
            
        # 2. Arithmetic Progressions with different steps
        # Small steps create more collisions in sums/products?
        # d=1 is min sums (2n-3=25). Larger d increases sum range.
        # But maybe larger d helps product collisions?
        for d in [2, 3]:
            # Ensure max <= 10000
            # start + 13*d <= 10000 -> start <= 10000 - 39
            for start in range(1, 20):
                vals = [start + i * d for i in range(14)]
                if max(vals) <= maximum:
                    sets.append(vals)
        
        # 3. Geometric-like / Exponential (might have unique sums, but maybe structured products?)
        # Powers of 2 have unique sums (binary). Bad for sums.
        # But let's try small bases.
        for base in [2, 3]:
            vals = []
            x = 1
            while len(vals) < 14:
                vals.append(x)
                x *= base
                if x > maximum: break
            if len(vals) == 14:
                sets.append(vals)
        
        # 4. Clustered values
        # e.g., {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14} is one cluster.
        # Try {1..10} U {20..23}?
        # This breaks the AP sum minimality, but might help products?
        # Unlikely to beat AP for sums, but worth checking a few.
        
        # 5. Random sets with small values
        random.seed(42)
        for _ in range(100):
            # Sample 14 distinct ints from a small range to encourage collisions
            # Range 1..20 gives max sum 39, max prod 380
            # Range 1..30
            limit = random.choice([15, 20, 25, 30, 40])
            vals = random.sample(range(1, limit + 1), 14)
            vals.sort()
            sets.append(vals)
            
        # 6. Perturbations of {1..14}
        base = list(range(1, 15))
        for _ in range(50):
            v = base.copy()
            # Swap 1-2 elements with nearby values
            idx1 = random.randint(0, 13)
            idx2 = random.randint(0, 13)
            if idx1 == idx2: continue
            # Try to replace one value with another small int not in set
            # Or just jitter
            v[idx1] += random.randint(-2, 2)
            v[idx2] += random.randint(-2, 2)
            if len(set(v)) == 14 and all(x > 0 for x in v) and max(v) <= 100:
                v.sort()
                sets.append(v)

        # Deduplicate
        unique_sets = []
        seen = set()
        for s in sets:
            key = tuple(s)
            if key not in seen:
                seen.add(key)
                unique_sets.append(s)
        return unique_sets

    def optimize_edges_sa(values):
        """
        Uses Simulated Annealing to find edges minimizing max(distinct_sums, distinct_prods).
        """
        n = len(values)
        all_pairs = list(combinations(range(n), 2))
        
        # Pre-calculate sum and product for all pairs
        pair_data = {}
        for i, j in all_pairs:
            pair_data[(i, j)] = (values[i] + values[j], values[i] * values[j])
        
        # Initial solution: Greedy
        # Sort pairs by frequency of product (higher is better for reuse)
        prod_counts = {}
        for p, (s, pr) in pair_data.items():
            prod_counts[pr] = prod_counts.get(pr, 0) + 1
        
        # Sort by product count desc, then sum count desc?
        # Let's just pick random 35 or greedy by product count.
        # Greedy: pick edge with highest product count, remove, repeat.
        # This is O(m * E) which is fine.
        
        def greedy_init():
            edges = []
            used = set()
            remaining = all_pairs[:]
            # Sort remaining by prod_count
            # We can't just sort once because removing an edge doesn't change product counts of others, 
            # but it does change which sums/prods are "new". 
            # Simple greedy: at each step, pick edge that adds fewest new sums/prods.
            cur_sums = set()
            cur_prods = set()
            
            for _ in range(m):
                best_edge = None
                best_cost = float('inf')
                for p in remaining:
                    if p in used: continue
                    s, pr = pair_data[p]
                    cost = (0 if s in cur_sums else 1) + (0 if pr in cur_prods else 1)
                    # Tie break: prefer higher product frequency?
                    if cost < best_cost or (cost == best_cost and random.random() < 0.5):
                        best_cost = cost
                        best_edge = p
                if best_edge is None: break
                edges.append(best_edge)
                used.add(best_edge)
                s, pr = pair_data[best_edge]
                cur_sums.add(s)
                cur_prods.add(pr)
            return edges

        current_edges = greedy_init()
        cur_s, cur_p = _evaluate_edges(values, current_edges)
        current_score = max(cur_s, cur_p)
        
        # Simulated Annealing
        # State: set of edges
        # Move: 
        #   1. Remove a random edge, add a random unselected edge.
        #   2. (Optional) Add/remove if size not m? No, size is fixed m.
        
        # To speed up, we maintain sums and prods counts (multisets)
        sum_counts = {}
        prod_counts_edges = {} # counts of how many edges have this product
        
        def build_counts(edges):
            sc = {}
            pc = {}
            for e in edges:
                s, p = pair_data[e]
                sc[s] = sc.get(s, 0) + 1
                pc[p] = pc.get(p, 0) + 1
            return sc, pc

        sum_counts, prod_counts_edges = build_counts(current_edges)
        
        # Temperature schedule
        t = 50.0
        cooling = 0.95
        min_t = 0.1
        max_iters = 50000 # Adjust based on time
        
        best_edges = current_edges[:]
        best_score = current_score
        best_s, best_p = cur_s, cur_p
        
        # Pre-compute list of edges for random selection
        edge_indices = {e: i for i, e in enumerate(all_pairs)}
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if t < min_t:
                break
                
            # Pick random edge to remove
            idx_rm = random.randint(0, len(current_edges) - 1)
            e_rm = current_edges[idx_rm]
            
            # Pick random edge to add (not in current)
            # To avoid O(N) search for random unselected, maintain a set of selected
            selected_set = set(current_edges)
            # Random retry to find an unselected edge
            e_add = None
            attempts = 0
            while e_add is None:
                e_candidate = random.choice(all_pairs)
                if e_candidate not in selected_set:
                    e_add = e_candidate
                attempts += 1
                if attempts > 20:
                    # Fallback: iterate
                    for e in all_pairs:
                        if e not in selected_set:
                            e_add = e
                            break
                    if e_add is None: break
                    
            if e_add is None:
                continue
                
            # Calculate change in score
            # Remove e_rm
            s_rm, p_rm = pair_data[e_rm]
            s_add, p_add = pair_data[e_add]
            
            # New counts simulation
            # sum_counts[s_rm] decreases
            # prod_counts_edges[p_rm] decreases
            
            # We need new distinct counts
            # dist_sums = len([k for k, v in sum_counts.items() if v > 0])
            # This is slow if done from scratch. 
            # Let's track number of distinct sums and prods.
            
            # Actually, maintaining exact distinct counts with dynamic updates is tricky.
            # Given m=35 and N=91, recalculating from scratch is ~35 ops.
            # 50000 iters * 35 ops = 1.75M ops. Very fast.
            # Let's just recalc.
            
            new_edges = current_edges[:]
            new_edges[idx_rm] = e_add
            
            new_s, new_p = _evaluate_edges(values, new_edges)
            new_score = max(new_s, new_p)
            
            # Acceptance
            delta = new_score - current_score
            if delta < 0 or (t > 0 and random.random() < pow(2.718, -delta / t)):
                current_edges = new_edges
                current_score = new_score
                cur_s, cur_p = new_s, new_p
                if current_score < best_score:
                    best_score = current_score
                    best_edges = current_edges[:]
                    best_s, best_p = cur_s, cur_p
            
            t *= cooling
            
        return best_edges

    value_sets = generate_value_sets()
    random.shuffle(value_sets)
    
    best_global_result = None
    best_global_score = float('inf')
    
    # Process value sets
    # To ensure we try the most promising ones (small values) first, we might sort by max value
    value_sets.sort(key=lambda x: max(x))
    
    for vals in value_sets:
        if time.time() > deadline:
            break
        
        edges = optimize_edges_sa(vals)
        score = _raw_value(vals, edges)
        
        if score < best_global_score:
            best_global_score = score
            best_global_result = {'values': vals, 'edges': [list(e) for e in edges]}
            
            # Early stopping if we hit theoretical minimum?
            # Min sums is 25. If score is 25, we can't do better.
            if best_global_score <= 25:
                break

    # Fallback if something went wrong
    if best_global_result is None:
        vals = list(range(1, 15))
        edges = optimize_edges_sa(vals)
        best_global_result = {'values': vals, 'edges': [list(e) for e in edges]}
        
    # Final Type Enforcement
    best_global_result['values'] = [int(v) for v in best_global_result['values']]
    best_global_result['edges'] = [[int(i), int(j)] for i, j in best_global_result['edges']]
    
    return best_global_result
# EVOLVE-BLOCK-END
