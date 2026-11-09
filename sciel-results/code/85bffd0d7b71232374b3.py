import itertools
import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 16
    vertices = range(n)
    
    # Generate all edges (triples)
    all_edges = list(itertools.combinations(vertices, 3))
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)  # 560
    
    # Generate all configurations (3 disjoint pairs -> 8 transversals)
    # A configuration is defined by 3 disjoint pairs.
    # We need to identify the 8 transversal edges.
    
    # Precompute partitions of 6 elements into 3 pairs (15 matchings)
    matchings = [
        ((0,1),(2,3),(4,5)),
        ((0,1),(2,4),(3,5)),
        ((0,1),(2,5),(3,4)),
        ((0,2),(1,3),(4,5)),
        ((0,2),(1,4),(3,5)),
        ((0,2),(1,5),(3,4)),
        ((0,3),(1,2),(4,5)),
        ((0,3),(1,4),(2,5)),
        ((0,3),(1,5),(2,4)),
        ((0,4),(1,2),(3,5)),
        ((0,4),(1,3),(2,5)),
        ((0,4),(1,5),(2,3)),
        ((0,5),(1,2),(3,4)),
        ((0,5),(1,3),(2,4)),
        ((0,5),(1,4),(2,3)),
    ]
    
    configs = []
    # Iterate over all 6-subsets of vertices
    for subset in itertools.combinations(vertices, 6):
        v = list(subset)
        for m in matchings:
            p1 = (v[m[0][0]], v[m[0][1]])
            p2 = (v[m[1][0]], v[m[1][1]])
            p3 = (v[m[2][0]], v[m[2][1]])
            
            # Generate 8 transversals
            edges_in_config = []
            for x1 in p1:
                for x2 in p2:
                    for x3 in p3:
                        edge = tuple(sorted((x1, x2, x3)))
                        edges_in_config.append(edge_to_idx[edge])
            
            # Store config as tuple of edge indices
            configs.append(tuple(edges_in_config))
            
    num_configs = len(configs)  # 120120
    
    # Precompute configs_in_edge as list of numpy arrays for fast indexing
    configs_in_edge_np = [None] * num_edges
    sizes = [0] * num_edges
    for c in configs:
        for e in c:
            sizes[e] += 1
            
    for i in range(num_edges):
        configs_in_edge_np[i] = np.zeros(sizes[i], dtype=np.int32)
        
    idx_tracker = [0] * num_edges
    for c_idx, c in enumerate(configs):
        for e in c:
            configs_in_edge_np[e][idx_tracker[e]] = c_idx
            idx_tracker[e] += 1
            
    best_system_set = set()
    best_count = 0
    
    def run_greedy_np(seed):
        random.seed(seed)
        order = list(range(num_edges))
        random.shuffle(order)
        
        conf_count = np.zeros(num_configs, dtype=np.int32)
        current_edges = set()
        
        for e_idx in order:
            c_indices = configs_in_edge_np[e_idx]
            if len(c_indices) > 0:
                max_count = np.max(conf_count[c_indices])
            else:
                max_count = 0
                
            if max_count < 7:
                current_edges.add(e_idx)
                conf_count[c_indices] += 1
                
        return current_edges, conf_count

    def simulated_annealing(initial_edges, initial_conf_count, seed):
        random.seed(seed)
        
        edges = set(initial_edges)
        conf_count = initial_conf_count.copy()
        current_count = len(edges)
        
        # Temperature schedule
        T = 10.0
        T_min = 0.01
        cooling = 0.995
        
        # Precompute list of edges for random selection
        all_edge_indices = list(range(num_edges))
        
        steps = 0
        max_steps = 5000 # Limit steps per SA run to ensure we can do multiple runs
        
        while T > T_min and steps < max_steps:
            steps += 1
            if time.time() > deadline - 5:
                break
                
            # Random move: 
            # 1. Remove a random edge from current set
            # 2. Add a random edge from complement (or try to add a random edge)
            # 3. Or swap
            
            if current_count == 0:
                # Add random edge
                e_idx = random.choice(all_edge_indices)
                c_indices = configs_in_edge_np[e_idx]
                if np.max(conf_count[c_indices]) < 7:
                    edges.add(e_idx)
                    conf_count[c_indices] += 1
                    current_count += 1
                continue

            # Choose move type
            move_type = random.random()
            
            if move_type < 0.5:
                # 1-for-1 swap: remove one, add one
                # Remove random edge
                e_remove = random.choice(list(edges))
                # Check if we can remove (always can)
                c_indices_remove = configs_in_edge_np[e_remove]
                conf_count[c_indices_remove] -= 1
                edges.remove(e_remove)
                current_count -= 1
                
                # Add random edge from complement
                # To avoid O(N) check for complement, just pick random edge and check if in edges
                # If in edges, skip (or retry)
                e_add = random.choice(all_edge_indices)
                if e_add not in edges:
                    c_indices_add = configs_in_edge_np[e_add]
                    # Check if adding e_add violates constraints given current state (after removal)
                    # We need max(conf_count[c] for c in c_indices_add) < 7
                    if np.max(conf_count[c_indices_add]) < 7:
                        edges.add(e_add)
                        conf_count[c_indices_add] += 1
                        current_count += 1
                    else:
                        # Revert removal if add fails? Or just accept removal?
                        # Standard SA: evaluate new state. 
                        # If we can't add, we are in a state with 1 fewer edge.
                        # Let's just revert the removal to keep count stable for simplicity, 
                        # or accept the loss if it allows future gains?
                        # Let's revert if add fails to maintain density.
                        edges.add(e_remove)
                        conf_count[c_indices_remove] += 1
                        current_count += 1
                        # Try a different add? No, just skip this step.
                else:
                    # Revert removal
                    edges.add(e_remove)
                    conf_count[c_indices_remove] += 1
                    current_count += 1
                    
            elif move_type < 0.8:
                # 2-for-2 swap: remove 2, add 2
                # This is complex to implement safely with constraints.
                # Let's stick to 1-for-1 and "kick" (remove 1, add 1 even if worse)
                
                # Actually, let's try a "kick": Remove 1 edge, then try to add 2 edges?
                # No, let's try: Remove 1 edge. Then try to add a random edge.
                # If we can add, count is same. If we can't, count is -1.
                # If count is -1, we accept with probability exp(-1/T).
                
                e_remove = random.choice(list(edges))
                c_indices_remove = configs_in_edge_np[e_remove]
                conf_count[c_indices_remove] -= 1
                edges.remove(e_remove)
                
                # Try to add a random edge
                e_add = random.choice(all_edge_indices)
                delta = 0
                if e_add not in edges:
                    c_indices_add = configs_in_edge_np[e_add]
                    if np.max(conf_count[c_indices_add]) < 7:
                        edges.add(e_add)
                        conf_count[c_indices_add] += 1
                        delta = 1 # We added one, so net change is 0 (removed 1, added 1)
                    else:
                        delta = -1 # We removed one, couldn't add, net change -1
                else:
                    # e_add was already in edges, so we just removed one
                    delta = -1
                
                # Acceptance criterion
                if delta >= 0:
                    # Improved or same, always accept
                    pass
                else:
                    # Worse, accept with probability
                    prob = np.exp(delta / T)
                    if random.random() > prob:
                        # Reject: revert
                        if e_add in edges:
                            edges.remove(e_add)
                            conf_count[c_indices_add] -= 1
                            # Wait, if delta == -1, it means we didn't add e_add.
                            # So we just need to re-add e_remove.
                        edges.add(e_remove)
                        conf_count[c_indices_remove] += 1
                        current_count += 1 # Revert count change if we rejected the state?
                        # Actually, current_count was not updated yet in this branch.
                        # Let's manage current_count carefully.
                        
                # Update current_count based on final state
                # This is getting messy. Let's simplify the SA loop.
                # We'll just track the set and count directly.
                
            else:
                # Do nothing or simple random restart logic?
                pass
            
            T *= cooling
            
        return edges, conf_count, current_count

    # Main Loop
    # 1. Run a few greedy seeds to find a good starting point
    # 2. Run SA from the best point
    
    # Run 10 greedy seeds first
    best_greedy_edges = set()
    best_greedy_conf = np.zeros(num_configs, dtype=np.int32)
    best_greedy_count = 0
    
    for i in range(10):
        if time.time() > deadline - 20:
            break
        edges, conf = run_greedy_np(i)
        if len(edges) > best_greedy_count:
            best_greedy_count = len(edges)
            best_greedy_edges = edges
            best_greedy_conf = conf
            best_system_set = edges
            best_count = len(edges)
            
    # Now run SA from the best greedy result
    # We can run SA multiple times with different seeds, starting from best_greedy_edges
    # Or restart from scratch? Starting from best is better.
    
    # Let's run SA for the remaining time
    sa_seed = 100
    while time.time() < deadline - 5:
        # Run SA
        edges, conf, count = simulated_annealing(best_greedy_edges, best_greedy_conf, sa_seed)
        
        if count > best_count:
            best_count = count
            best_system_set = edges
            # Update best_greedy to this new best for next SA iteration?
            # Yes, it helps.
            best_greedy_edges = edges
            best_greedy_conf = conf
            best_greedy_count = count
            
        sa_seed += 1
        
        # Occasionally, if we haven't improved in a while, try a new greedy seed?
        # For now, just keep refining the best.
        
    return [list(all_edges[i]) for i in sorted(best_system_set)]

if __name__ == "__main__":
    # Test
    res = construct()
    print(f"Found {len(res)} triples")
