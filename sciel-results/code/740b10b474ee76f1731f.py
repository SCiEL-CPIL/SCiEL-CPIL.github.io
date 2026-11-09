import time
import random
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    trees = instance_trees()
    N_HOSTS = 20
    
    # Pre-process trees
    # trees[i] is for T_{i+1}
    
    # State: list of 20 entries. state[i] is None or a list of k host vertices for T_{i+1}.
    # We will use Simulated Annealing to find a packing with minimum conflicts.
    
    # Initialize with random embeddings
    state = [None] * 20
    for i in range(20):
        k = i + 1
        if k == 1:
            state[i] = [random.randint(0, 19)]
        else:
            # Random permutation of 20 hosts, take first k
            hosts = list(range(N_HOSTS))
            random.shuffle(hosts)
            state[i] = hosts[:k]
    
    # Calculate conflicts
    # Conflict is a pair of trees that share an edge in K20.
    # We want to minimize the number of conflicts.
    # Total edges in K20 is 190.
    # Total edges in trees is sum(k-1) for k=1..20 = 190.
    # So a perfect packing has 0 conflicts.
    
    def calc_conflicts(state):
        used_edges = set()
        conflicts = 0
        for i in range(20):
            k = i + 1
            if state[i] is None:
                continue
            mapping = state[i]
            if k == 1:
                continue
            tree_edges = trees[i]
            for u, v in tree_edges:
                e1 = min(mapping[u], mapping[v])
                e2 = max(mapping[u], mapping[v])
                if (e1, e2) in used_edges:
                    conflicts += 1
                else:
                    used_edges.add((e1, e2))
        return conflicts
    
    current_conflicts = calc_conflicts(state)
    best_state = [s[:] if s else None for s in state]
    best_conflicts = current_conflicts
    
    temperature = 1.0
    cooling = 0.99
    
    while time.time() < deadline:
        if current_conflicts == 0:
            break
            
        # Choose a random tree to modify
        # Weighted by size? Larger trees have more impact.
        # Let's just pick uniformly first.
        i = random.randint(0, 19)
        k = i + 1
        if k == 1:
            continue
            
        old_mapping = state[i]
        
        # Modification strategies:
        # 1. Swap two vertices in the mapping
        # 2. Move a vertex to a random host (if not occupied by another vertex in same tree)
        # 3. Random restart for this tree (random permutation)
        
        strategy = random.random()
        
        if strategy < 0.3:
            # Swap
            new_mapping = old_mapping[:]
            a, b = random.sample(range(k), 2)
            new_mapping[a], new_mapping[b] = new_mapping[b], new_mapping[a]
        elif strategy < 0.6:
            # Move
            new_mapping = old_mapping[:]
            a = random.randint(0, k - 1)
            # Find a host not in new_mapping
            used_hosts = set(new_mapping)
            available = [h for h in range(N_HOSTS) if h not in used_hosts]
            if available:
                new_mapping[a] = random.choice(available)
            else:
                # No available host, just swap with another
                b = random.randint(0, k - 1)
                new_mapping[a], new_mapping[b] = new_mapping[b], new_mapping[a]
        else:
            # Random restart for this tree
            hosts = list(range(N_HOSTS))
            random.shuffle(hosts)
            new_mapping = hosts[:k]
            
        # Calculate delta
        # Efficient delta calculation is complex.
        # For now, recalculate if k is small or if we are close to 0.
        # Or just recalculate every time if it's fast enough.
        # 20 trees, max 19 edges each. 380 edges total.
        # Recalculating is O(20 * 19) ~ 400 ops. Very fast.
        
        state[i] = new_mapping
        new_conflicts = calc_conflicts(state)
        
        delta = new_conflicts - current_conflicts
        
        if delta < 0 or random.random() < pow(2.718, -delta / temperature):
            current_conflicts = new_conflicts
            if current_conflicts < best_conflicts:
                best_conflicts = current_conflicts
                best_state = [s[:] if s else None for s in state]
        else:
            state[i] = old_mapping
            
        temperature *= cooling
        if temperature < 1e-4:
            temperature = 1e-4

    # Return the best state found
    return best_state
