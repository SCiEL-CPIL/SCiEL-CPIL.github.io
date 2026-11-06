import random
import time
import math

def construct():
    n = 16
    
    # Pre-compute all possible triples
    all_triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                all_triples.append((i, j, k))
                
    # Function to calculate exact independence number
    def calc_alpha(triples):
        if not triples:
            return n
        triple_masks = [ (1<<a) | (1<<b) | (1<<c) for a,b,c in triples ]
        max_size = 0
        for mask in range(1 << n):
            is_indep = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    is_indep = False
                    break
            if is_indep:
                size = bin(mask).count('1')
                if size > max_size:
                    max_size = size
        return max_size

    # Fast approximate independence number (Greedy)
    # Returns a lower bound on alpha.
    def calc_alpha_greedy(triples):
        if not triples:
            return n
        
        # Build adjacency list of vertices connected by triples
        # adj[u] = set of v such that u,v are in a triple
        adj = [set() for _ in range(n)]
        for a, b, c in triples:
            adj[a].add(b)
            adj[a].add(c)
            adj[b].add(a)
            adj[b].add(c)
            adj[c].add(a)
            adj[c].add(b)
            
        available = set(range(n))
        size = 0
        # Randomize the order of picking to get a better estimate
        # We pick a random vertex, add it, remove it and its neighbors
        while available:
            # Pick a random vertex
            v = random.choice(list(available))
            size += 1
            # Remove v and its neighbors
            to_remove = {v} | adj[v]
            available -= to_remove
            
        return size

    # Initial random linear hypergraph
    # Greedy construction
    def generate_greedy():
        cur_triples = []
        used_pairs_set = set()
        
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        
        for t in shuffled:
            a, b, c = t
            pairs = [(a,b), (a,c), (b,c)]
            conflict = False
            for p in pairs:
                if p in used_pairs_set:
                    conflict = True
                    break
            if not conflict:
                cur_triples.append(t)
                used_pairs_set.update(pairs)
        return cur_triples

    best_alpha = n
    best_config = []
    
    start_time = time.time()
    time_limit = 110 # 110 seconds to be safe

    # Run multiple restarts with SA
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # 1. Generate initial solution
        cur_triples = generate_greedy()
        
        # 2. Calculate exact alpha for initial
        cur_alpha_exact = calc_alpha(cur_triples)
        if cur_alpha_exact < best_alpha:
            best_alpha = cur_alpha_exact
            best_config = cur_triples[:]
            
        # 3. Simulated Annealing
        # State: cur_triples, used_pairs
        used_pairs = set()
        for t in cur_triples:
            a,b,c = t
            used_pairs.add((a,b))
            used_pairs.add((a,c))
            used_pairs.add((b,c))
            
        # Proxy cost
        cur_cost = calc_alpha_greedy(cur_triples)
        
        T = 1.0
        cooling = 0.99
        min_T = 0.01
        
        steps = 0
        max_steps = 500 # Limit steps per restart to allow multiple restarts
        
        while T > min_T and steps < max_steps:
            if time.time() - start_time > time_limit:
                break
                
            steps += 1
            
            # Generate a move
            # 50% Add, 50% Swap
            move_type = random.random()
            
            new_triples = cur_triples[:]
            new_used_pairs = used_pairs.copy()
            valid_move = False
            
            if move_type < 0.5:
                # Try to add a random triple
                t_new = random.choice(all_triples)
                a,b,c = t_new
                pairs = [(a,b), (a,c), (b,c)]
                if not any(p in new_used_pairs for p in pairs):
                    new_triples.append(t_new)
                    new_used_pairs.update(pairs)
                    valid_move = True
            else:
                # Try to swap: remove one, add one
                if cur_triples:
                    t_out_idx = random.randint(0, len(cur_triples)-1)
                    t_out = cur_triples[t_out_idx]
                    a_out, b_out, c_out = t_out
                    pairs_out = [(a_out,b_out), (a_out,c_out), (b_out,c_out)]
                    
                    # Remove t_out
                    new_triples.pop(t_out_idx)
                    for p in pairs_out:
                        new_used_pairs.discard(p)
                        
                    # Try to add a random triple
                    t_new = random.choice(all_triples)
                    if t_new != t_out:
                        a_in, b_in, c_in = t_new
                        pairs_in = [(a_in,b_in), (a_in,c_in), (b_in,c_in)]
                        if not any(p in new_used_pairs for p in pairs_in):
                            new_triples.append(t_new)
                            new_used_pairs.update(pairs_in)
                            valid_move = True
                    # If not valid, revert
                    if not valid_move:
                         new_triples.insert(t_out_idx, t_out)
                         new_used_pairs.update(pairs_out)
                         valid_move = False # Mark as invalid move to skip

            if valid_move:
                # Evaluate new cost
                new_cost = calc_alpha_greedy(new_triples)
                delta = new_cost - cur_cost
                
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    # Accept
                    cur_triples = new_triples
                    used_pairs = new_used_pairs
                    cur_cost = new_cost
                    
                    # Occasionally check exact alpha
                    if steps % 10 == 0 or delta < 0:
                         exact_a = calc_alpha(cur_triples)
                         if exact_a < best_alpha:
                             best_alpha = exact_a
                             best_config = cur_triples[:]
                             # Reset temperature? Or just continue.
                             # If we found a new best, we might want to restart SA from here?
                             # For now, just continue.
                    
            T *= cooling

    return [list(t) for t in best_config]
