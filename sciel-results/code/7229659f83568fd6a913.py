import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    D = 11
    rng = np.random.RandomState(42)
    
    # --- 1. Construct Type A Vectors ---
    # 22 vectors: +/- 2 * e_i
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
    
    # --- 2. Define Support Graph for Type D ---
    # Type D vectors have support of size 4.
    # There are C(11, 4) = 330 possible supports.
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)
    
    # Precompute bitmasks for supports for fast intersection checks
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    
    # Build conflict graph: conflict[i][j] = True if supports i and j intersect in >= 3 elements
    # Intersection size >= 3 implies max dot product can be 3 or 4, leading to dist < 2.
    # Intersection size <= 2 implies max dot product is 2, leading to dist >= 2.
    # We need an Independent Set in this graph.
    
    # Adjacency list for faster neighbor checks
    neighbors = [[] for _ in range(N_S)]
    for i in range(N_S):
        mi = support_masks[i]
        for j in range(i + 1, N_S):
            mj = support_masks[j]
            inter = mi & mj
            if inter.bit_count() >= 3:
                neighbors[i].append(j)
                neighbors[j].append(i)
    
    # --- 3. Simulated Annealing for Maximum Independent Set ---
    
    def random_greedy_init():
        """Generate a valid independent set using greedy randomization."""
        selected = set()
        available = set(range(N_S))
        while available:
            # Pick a random node from available
            v = rng.choice(list(available))
            selected.add(v)
            # Remove v and its neighbors from available
            available.difference_update(neighbors[v])
            available.discard(v)
        return selected

    # Initial best solution
    best_set = random_greedy_init()
    best_size = len(best_set)
    
    # Simulated Annealing Parameters
    # We want to explore the space of independent sets.
    # Moves: Add, Remove, Swap.
    # To keep validity, we track 'blocked' nodes (neighbors of selected nodes).
    
    # We will run multiple SA chains or a long single chain.
    # Given N_S=330, we can afford many steps.
    
    # State representation for speed:
    # x: set of selected nodes
    # blocked: set of nodes that cannot be added (neighbors of x)
    
    def compute_blocked(x):
        blocked = set()
        for v in x:
            blocked.update(neighbors[v])
        blocked.difference_update(x) # Selected nodes are not "blocked" in the sense of being addable, 
                                     # but they are in x. We just need to know what's not in x and not blocked.
        return blocked
    
    current_x = set(best_set)
    current_blocked = compute_blocked(current_x)
    current_size = len(current_x)
    
    T_start = 10.0
    T_end = 0.01
    alpha = 0.999 # Cooling rate
    # Estimate steps: We have ~140s. Let's aim for 2-5 million steps if fast enough.
    # Python sets are fast.
    
    steps = 0
    max_steps = 2000000 # Cap to prevent infinite loops if time check fails, though time check is primary
    
    while time.time() < deadline and steps < max_steps:
        T = T_start * (T_end / T_start) ** (steps / max_steps)
        
        # Choose a move type
        move_type = rng.random()
        
        if move_type < 0.3:
            # Add a node
            # Candidates: nodes not in current_x and not in current_blocked
            # We can't easily iterate "not in x and not in blocked" efficiently without a set difference.
            # Let's maintain a set of 'free' nodes? Or just sample.
            # Sampling: Pick random node, check if free.
            # Better: Maintain a list of free nodes? Updating it is O(N) per step.
            # Let's just sample from all nodes and check.
            # N_S is small (330).
            
            # Optimization: Precompute list of all nodes.
            # If we pick a node v:
            # If v in current_x: invalid add.
            # If v in current_blocked: invalid add.
            # Else: valid add.
            
            # To make this efficient, we can try to pick from a subset.
            # But simple random sampling works if we accept/reject.
            
            # Let's try a different approach for SA moves that are always valid:
            # 1. Swap: Pick v in current_x, u not in current_x.
            #    If u is not neighbor of (current_x - {v}), then swap is valid.
            # 2. Add: Pick u not in current_x, not in current_blocked.
            # 3. Remove: Pick v in current_x.
            
            # Let's stick to Swap and Add/Remove.
            
            if move_type < 0.4: # 0.1 probability for Add
                # Try to add a node
                # Pick random candidate from all nodes
                # To avoid bias, we could pick from nodes not in x.
                # Let's just pick a random node v.
                v = rng.randint(N_S)
                if v in current_x or v in current_blocked:
                    continue # Invalid move, skip (or count as step)
                
                # Proposed: current_x U {v}
                # Delta size = +1
                # Always accept if valid? No, SA accepts worse states too, but size is the objective.
                # Here, adding a node increases size. So always accept?
                # Wait, adding a node might block future additions.
                # But we are maximizing size.
                # If we accept all valid adds, we just do greedy.
                # SA needs to accept *removals* or *swaps* that might lead to larger sets later.
                # Actually, for Max Independent Set, "Add" is always an improvement in objective.
                # The challenge is getting stuck in local optima where you can't add any more, 
                # but swapping two nodes allows adding three more.
                # So "Add" is greedy. "Swap" is the key to escape local optima.
                
                # Let's prioritize Swaps.
                pass # Handled below in logic restructuring
                
            # Let's restructure moves:
            # 1. Swap (50%): Pick v in x, u not in x. If u compatible with x\{v}, swap.
            # 2. Remove (20%): Pick v in x, remove it. (Accept with prob exp(-1/T) since it decreases size)
            # 3. Add (30%): Pick u not in x, not blocked. Add it. (Always accept, increases size)
            
            # Re-evaluating move selection
            r = rng.random()
            if r < 0.5:
                # Swap
                if not current_x: continue
                v = rng.choice(list(current_x))
                # Try to find a valid u to swap with
                # u must not be in x
                # u must not be neighbor of any node in (x - {v})
                
                # Optimization: Instead of random u, we can look at neighbors of v?
                # No, u can be anywhere.
                # Let's sample u.
                # If we sample u randomly from 1..N_S-1, we might hit x often.
                
                # Let's just try a random u.
                u = rng.randint(N_S)
                if u in current_x:
                    continue
                
                # Check if u is compatible with x \ {v}
                # u is compatible if no neighbor of u is in (x - {v})
                # i.e., (neighbors[u] & (current_x - {v})) is empty
                
                # Efficient check:
                # We have current_blocked which is neighbors of ALL x.
                # If u is not in current_blocked, it's compatible with ALL x.
                # If u IS in current_blocked, it might still be compatible with x \ {v} if the only conflict was with v.
                
                if u not in current_blocked:
                    # u is compatible with all x.
                    # Swap v with u.
                    # New x = x - {v} + {u}
                    # Size change = 0.
                    # Always accept? Or SA?
                    # Since size doesn't change, we accept with prob 1 to explore.
                    # Update sets.
                    current_x.remove(v)
                    current_x.add(u)
                    
                    # Update blocked
                    # Nodes that were only blocked by v are now free.
                    # Nodes that are neighbors of u are now blocked.
                    
                    # Recalculate blocked? O(N * deg) ~ 330 * 100 = 33000 ops.
                    # 200k steps * 33k ops = 6.6e9 ops. Too slow in Python.
                    
                    # We need incremental update or faster blocked check.
                    # Let's maintain a count of how many selected nodes block each node.
                    # block_count[w] = number of nodes v in x such that w in neighbors[v].
                    # A node w is blocked if block_count[w] > 0.
                    
                    # Let's restart with this data structure.
                    
                    # Abort this loop iteration, restart logic with block_counts.
                    break # Break to restart inner loop with better structure? No, let's just rewrite the SA loop properly below.
                    
            elif r < 0.8:
                # Remove
                if not current_x: continue
                v = rng.choice(list(current_x))
                # Remove v
                # Size change = -1
                # Accept prob = exp(-1/T)
                if rng.random() < np.exp(-1.0 / T):
                    current_x.remove(v)
                    # Update blocked
                    # This is getting complicated to do incrementally in Python without significant overhead.
                    pass
            else:
                # Add
                # Find a free node.
                # Free nodes: not in x, block_count == 0.
                pass

    # --- Re-implementing SA with Efficient Data Structures ---
    
    # Reset
    current_x = set(best_set)
    # block_count[i] = number of neighbors of i that are in current_x
    block_count = np.zeros(N_S, dtype=np.int32)
    for v in current_x:
        for nb in neighbors[v]:
            block_count[nb] += 1
            
    # free_nodes: list of nodes i where i not in current_x and block_count[i] == 0
    # We can't maintain this list dynamically easily without O(N) updates.
    # But N=330 is small. We can just scan or sample.
    
    # Let's use a simple random sampling strategy for candidates, but validate with block_count.
    
    T = 1.0
    cooling = 0.9995
    min_T = 0.01
    
    # We will run until deadline
    while time.time() < deadline:
        
        if T < min_T:
            T = min_T
            
        # Choose move
        r = rng.random()
        
        if r < 0.6:
            # Swap: Pick v in x, u not in x.
            # Constraint: u must be compatible with x \ {v}.
            # Compatibility: block_count[u] must be 0 OR block_count[u] == 1 and the only blocker is v.
            
            if not current_x:
                continue
                
            v = rng.choice(list(current_x))
            
            # Try to find a valid u
            # We can sample u randomly.
            # Probability of picking a valid u might be low if graph is dense.
            # Let's try a few times? No, just one attempt per step for simplicity.
            
            u = rng.randint(N_S)
            if u in current_x:
                continue
            
            # Check compatibility of u with x \ {v}
            # block_count[u] counts blockers in x.
            # If block_count[u] == 0: u is compatible with all x. Valid swap.
            # If block_count[u] == 1: We need to check if the single blocker is v.
            #    If yes, valid. If no, invalid.
            # If block_count[u] > 1: Invalid (at least one blocker is not v).
            
            valid_swap = False
            if block_count[u] == 0:
                valid_swap = True
            elif block_count[u] == 1:
                # Check if v is the only neighbor of u in x
                # We can check if v is in neighbors[u]
                # And if there are no other neighbors of u in x.
                # Since block_count[u] is 1, if v is a neighbor, it MUST be the only one.
                if v in neighbors[u]:
                    valid_swap = True
            # else invalid
            
            if not valid_swap:
                continue
                
            # Perform Swap
            # Remove v, Add u
            current_x.remove(v)
            current_x.add(u)
            
            # Update block_counts
            # v was blocking its neighbors. Remove v's contribution.
            for nb in neighbors[v]:
                block_count[nb] -= 1
            # u is blocking its neighbors. Add u's contribution.
            for nb in neighbors[u]:
                block_count[nb] += 1
                
            # Size unchanged. Accept always? 
            # In SA, we accept worse states. Here size is same.
            # We accept to explore.
            
        elif r < 0.8:
            # Remove: Pick v in x, remove it.
            if not current_x:
                continue
            v = rng.choice(list(current_x))
            
            # Delta size = -1
            # Accept prob = exp(-1/T)
            if rng.random() < np.exp(-1.0 / T):
                current_x.remove(v)
                for nb in neighbors[v]:
                    block_count[nb] -= 1
                # Size decreased.
                # Update best if needed? No, we track best separately.
                
        else:
            # Add: Pick u not in x, not blocked.
            # We need to find a free node.
            # Sampling: Pick random u.
            # If u in x: invalid.
            # If block_count[u] > 0: invalid.
            # Else: valid.
            
            # To increase success rate, we can bias towards nodes with low block_count?
            # Or just sample.
            
            # Let's try to find a free node by sampling up to 5 times?
            # Or just 1 time. If it fails, step is wasted.
            # Given N=330, if many are free, we'll find one.
            
            u = rng.randint(N_S)
            if u in current_x or block_count[u] > 0:
                continue
            
            # Valid Add
            current_x.add(u)
            for nb in neighbors[u]:
                block_count[nb] += 1
                
            # Size increased. Always accept.
            
        # Update Best
        if len(current_x) > best_size:
            best_size = len(current_x)
            best_set = set(current_x)
            
        T *= cooling
        
    # --- 4. Final Greedy Fill ---
    # Ensure we didn't miss any trivial additions due to SA randomness
    current_x = set(best_set)
    block_count = np.zeros(N_S, dtype=np.int32)
    for v in current_x:
        for nb in neighbors[v]:
            block_count[nb] += 1
            
    # Iterate until no more additions
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Find all free nodes
        free_nodes = [i for i in range(N_S) if i not in current_x and block_count[i] == 0]
        if free_nodes:
            # Add all of them? Or one by one?
            # Adding all might block others? No, free nodes are not neighbors of each other?
            # Wait. If u and w are both free, they are not neighbors of any node in x.
            # Are u and w neighbors of each other?
            # If u and w are neighbors, adding u blocks w.
            # So we should add them one by one, updating block counts.
            # Or just pick one, add, update, repeat.
            
            # Greedy: Pick a free node, add it.
            # To be safe, pick the one with minimum degree among free nodes?
            # Or just random.
            u = rng.choice(free_nodes)
            current_x.add(u)
            for nb in neighbors[u]:
                block_count[nb] += 1
            changed = True
            
    best_set = current_x
    best_size = len(best_set)
    
    # --- 5. Construct Final Vectors ---
    # Type A
    vectors_A_arr = np.array(vectors_A, dtype=np.int64)
    
    # Type D: For each support in best_set, add all 16 sign vectors
    vectors_D = []
    for s in best_set:
        # s is a tuple of 4 indices
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            vectors_D.append(v)
            
    if vectors_D:
        vectors_D_arr = np.array(vectors_D, dtype=np.int64)
        final_vectors = np.vstack([vectors_A_arr, vectors_D_arr])
    else:
        final_vectors = vectors_A_arr
        
    return final_vectors

