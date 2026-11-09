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
    
    # Pre-process trees to get edges and vertex lists
    # trees[i] is for T_{i+1}
    # We want to pack T1..T20.
    # T1 is trivial (1 vertex). T2 is 1 edge.
    
    # State: A list of lists. state[i] is the list of host vertices for tree T_{i+1}.
    # If a tree is not embedded, state[i] is None.
    # To simplify SA, we will assume all trees are always "present" but we track conflicts.
    # However, the task allows None. 
    # Strategy: 
    # 1. Initialize with random embeddings for all trees.
    # 2. Calculate total conflicts (edge overlaps).
    # 3. Anneal: 
    #    - If we have conflicts, try to move a vertex of a conflicting tree to a new host to reduce conflicts.
    #    - If we have no conflicts, we have a valid packing. We can try to "improve" by ensuring we maximize the number of trees? 
    #    Actually, if we start with all 20 trees, and we find a conflict-free state, we have packed all 20.
    #    The raw value is sum of edges. T1..T20 have 0, 1, 2, ..., 19 edges. Sum = 190.
    #    If we cannot pack all, we want to pack the largest ones? 
    #    The score is raw_value / seed. 
    #    If we pack all 20, value = 190.
    #    If we pack 19 (drop T1), value = 190. (T1 has 0 edges).
    #    If we pack 19 (drop T2), value = 189.
    #    So we want to maximize the number of edges. This is equivalent to maximizing the sum of k-1 for embedded trees.
    #    Since we can always embed T1 (0 edges) anywhere, we should always include it.
    #    The challenge is embedding T2..T20.
    
    # Let's try to pack all T2..T20.
    # If we fail, we might drop the smallest ones (least edge contribution) or the ones causing most conflicts.
    
    # Initialize state
    # state[i] is a list of length (i+1) containing host vertices for T_{i+1}
    state = []
    for i in range(20):
        k = i + 1
        # Random permutation of 0..19 of length k
        hosts = list(range(N_HOSTS))
        random.shuffle(hosts)
        state.append(hosts[:k])
        
    # Pre-compute edges for each tree
    tree_edges = []
    for i in range(20):
        tree_edges.append(trees[i])
        
    # Function to calculate total conflicts and score
    # Conflicts: pairs of edges (u,v) from different trees that map to the same host edge (a,b).
    # We can represent a host edge as (min(a,b), max(a,b)).
    
    def get_edges_for_tree(i, mapping):
        edges = []
        for u, v in tree_edges[i]:
            a = mapping[u]
            b = mapping[v]
            if a == b:
                # Self-loop? Not possible in simple graph embedding if mapping is injective?
                # Wait, mapping is a list of distinct hosts?
                # The problem says "list of k distinct int host vertices".
                # So mapping must be injective.
                # My initialization uses distinct hosts.
                # Moves must preserve distinctness within a tree.
                pass
            e = (min(a, b), max(a, b))
            edges.append(e)
        return edges

    # To make SA fast, we maintain a count of edge usages.
    # edge_count[e] = number of trees using host edge e.
    # Conflict exists if edge_count[e] > 1.
    # Total conflicts = sum(max(0, count[e] - 1) for e in edge_count)
    
    # Initialize edge_count
    edge_count = {}
    for i in range(20):
        edges = get_edges_for_tree(i, state[i])
        for e in edges:
            edge_count[e] = edge_count.get(e, 0) + 1
            
    def calc_conflicts():
        conf = 0
        for e, c in edge_count.items():
            if c > 1:
                conf += c - 1
        return conf

    current_conflicts = calc_conflicts()
    
    # We also need to track which trees are "valid" (no internal conflicts? No, internal conflicts are impossible if mapping is injective).
    # The only constraint is edge-disjointness between trees.
    # So if total_conflicts == 0, we have a valid packing of all 20 trees.
    
    # If we can't get to 0, we might want to drop trees.
    # However, SA is good at finding 0 if it exists.
    # If 0 is not reachable, we want to minimize conflicts, but also maximize the "weight" of the embedded trees.
    # Actually, if we have conflicts, the solution is INVALID.
    # So we MUST reach 0 conflicts to get any score > 0?
    # "Any broken rule makes the whole result invalid".
    # So if we return a state with conflicts, score is 0.
    # We must return a state with 0 conflicts.
    # If we can't find one for all 20, we should return a subset.
    # Strategy: 
    # 1. Try to pack all 20.
    # 2. If stuck with conflicts after time, drop the tree with the fewest edges (T1, T2, etc.) and re-optimize?
    #    Or just return the best subset found that has 0 conflicts.
    
    # Let's run SA to minimize conflicts.
    # If we reach 0, we are done (Value = 190).
    # If we don't, we need a fallback.
    
    # To handle the "subset" case, we can track which trees are "active".
    # But changing the set of active trees is a complex move.
    # Simpler: 
    # Run SA on all 20.
    # If successful, return all.
    # If not, try removing T2 (1 edge) and SA on 19.
    # If not, remove T3, etc.
    
    # Given the deadline, we should prioritize finding a 0-conflict state for the largest possible subset.
    
    # Let's refine the SA move:
    # Pick a random tree i that is involved in a conflict.
    # Pick a random vertex v in that tree.
    # Move v to a new random host h' (distinct from other hosts in tree i).
    # Update edge_count and conflicts.
    
    # We need to know which trees are in conflict.
    # We can maintain a set of conflicting edges, and map edges to trees.
    # This is expensive to update perfectly.
    # Alternative: Just recalculate conflicts? 190 edges. 20 trees.
    # Recalculating is O(190). Very fast.
    
    def recalc_all():
        nonlocal edge_count, current_conflicts
        edge_count = {}
        for i in range(20):
            edges = get_edges_for_tree(i, state[i])
            for e in edges:
                edge_count[e] = edge_count.get(e, 0) + 1
        current_conflicts = 0
        for e, c in edge_count.items():
            if c > 1:
                current_conflicts += c - 1

    # SA Loop
    T = 1.0
    cooling = 0.9995
    best_state = [s[:] for s in state]
    best_conflicts = current_conflicts
    
    while time.time() < deadline:
        if current_conflicts == 0:
            # We have a valid packing of all 20.
            break
            
        # Pick a move
        # Bias towards trees with conflicts?
        # For now, random tree, random vertex.
        i = random.randint(0, 19)
        k = i + 1
        v = random.randint(0, k - 1)
        
        old_host = state[i][v]
        
        # Find new host
        # Must be distinct from other hosts in tree i
        used_in_tree = set(state[i])
        used_in_tree.remove(old_host)
        
        # Choose a random host not in used_in_tree
        # Optimization: pick from 0..19
        new_host = random.randint(0, 19)
        while new_host in used_in_tree:
            new_host = random.randint(0, 19)
            
        # Apply move
        state[i][v] = new_host
        
        # Recalculate conflicts
        old_conf = current_conflicts
        recalc_all()
        new_conf = current_conflicts
        
        delta = new_conf - old_conf
        
        if delta <= 0 or random.random() < pow(2.718, -delta / T):
            # Accept
            best_state = [s[:] for s in state]
            best_conflicts = new_conf
        else:
            # Reject
            state[i][v] = old_host
            current_conflicts = old_conf
            
        T *= cooling
        
    # After SA, check if we have a valid solution
    # If best_conflicts == 0, we have packed all 20.
    # If not, we need to drop trees.
    
    # Let's try to drop trees and see if we can get a valid packing.
    # We will try subsets in descending order of "value" (sum of edges).
    # Since T1 has 0 edges, dropping it doesn't change value.
    # T2 has 1 edge.
    # We want to maximize value.
    
    # If we failed to pack all 20, maybe we can pack 19?
    # We can try to force the state of the first 19 trees to be conflict-free, ignoring the 20th.
    # Or better: 
    # If we have conflicts, identify the set of trees involved in conflicts.
    # Try removing one of them and re-running a quick SA or checking if the rest is valid.
    
    # Simpler Fallback:
    # If we didn't reach 0 conflicts, try to find a valid subset.
    # We can iterate through possible subsets? No, too many.
    
    # Heuristic: 
    # The conflicts are likely localized.
    # Let's try to "fix" the current state by removing trees that are part of conflicts until it's valid.
    
    active_trees = list(range(20))
    # Sort by size ascending? Remove small ones first to preserve large edge count?
    # Actually, removing a large tree loses many edges.
    # So we should try to remove the SMALL trees first if they are causing conflicts.
    # But T1 and T2 are small.
    
    # Let's try a greedy removal from the current best_state.
    # Check if best_state is valid for all 20.
    # If not, try removing T1 (index 0). Check validity for 1..19.
    # If not, try removing T2 (index 1). Check validity for 0, 2..19.
    # This is O(20 * 190) checks.
    
    def is_valid_subset(indices):
        edge_count_subset = {}
        for i in indices:
            edges = get_edges_for_tree(i, best_state[i])
            for e in edges:
                edge_count_subset[e] = edge_count_subset.get(e, 0) + 1
        for e, c in edge_count_subset.items():
            if c > 1:
                return False
        return True

    # Try all 20
    if is_valid_subset(list(range(20))):
        result = [best_state[i][:] for i in range(20)]
        # Ensure types are int
        for i in range(20):
            result[i] = [int(x) for x in result[i]]
        return result

    # If not, try dropping trees one by one, starting from the smallest (least value loss)
    # But we must maintain a set of indices.
    # Let's try dropping T1 (0 edges).
    # Then T2 (1 edge).
    # Then T3 (2 edges).
    # etc.
    
    # We want to find the largest subset (by sum of edges) that is valid.
    # Since values are 0, 1, 2, ..., 19.
    # Dropping T_k loses k-1 edges.
    # We should try to drop the tree with the smallest "cost" that resolves the conflict.
    # But conflicts might involve multiple trees.
    
    # Simple approach: Try all subsets of size 19, 18, ...
    # There are 20 subsets of size 19.
    # 190 subsets of size 18.
    # 1140 subsets of size 17.
    # This is feasible.
    
    # However, we only have one candidate state (best_state) from SA.
    # It's possible that best_state is not valid for any large subset, but a different state would be.
    # But re-running SA for each subset is too slow.
    
    # Let's check if the current best_state works for any subset of size >= 15.
    # If not, we might need to do more SA or accept a lower score.
    
    # Let's try to repair best_state by removing conflicting trees.
    # Identify conflicts in best_state.
    
    def get_conflict_trees(indices):
        edge_count_subset = {}
        edge_to_trees = {}
        for i in indices:
            edges = get_edges_for_tree(i, best_state[i])
            for e in edges:
                edge_count_subset[e] = edge_count_subset.get(e, 0) + 1
                if e not in edge_to_trees:
                    edge_to_trees[e] = []
                edge_to_trees[e].append(i)
        
        conflicting_trees = set()
        for e, c in edge_count_subset.items():
            if c > 1:
                for t in edge_to_trees[e]:
                    conflicting_trees.add(t)
        return conflicting_trees

    current_indices = list(range(20))
    while True:
        if is_valid_subset(current_indices):
            break
        if not current_indices:
            break
            
        conf_trees = get_conflict_trees(current_indices)
        if not conf_trees:
            break
            
        # Remove the tree from conf_trees with the smallest edge count (k-1)
        # i.e., smallest index (since T_k has k-1 edges, index i has i edges)
        # Wait, index i corresponds to T_{i+1} which has i edges.
        # So T1 (idx 0) has 0 edges. T2 (idx 1) has 1 edge.
        # We want to remove the one with smallest i.
        
        min_idx = min(conf_trees)
        current_indices.remove(min_idx)
        
    # If we ended up with an empty set, we should at least return T1 (which is always valid)
    if not current_indices:
        current_indices = [0]
        
    # Construct result
    result = [None] * 20
    for i in current_indices:
        result[i] = [int(x) for x in best_state[i]]
        
    return result

# Test locally
if __name__ == "__main__":
    res = construct()
    # Check validity
    used = set()
    total_edges = 0
    trees = instance_trees()
    for i in range(20):
        if res[i] is None:
            continue
        mapping = res[i]
        # Check distinctness
        if len(set(mapping)) != len(mapping):
            print(f"Tree {i} not distinct")
        for u, v in trees[i]:
            a = mapping[u]
            b = mapping[v]
            e = (min(a, b), max(a, b))
            if e in used:
                print(f"Conflict at tree {i}, edge {e}")
            used.add(e)
            total_edges += 1
    print(f"Total edges: {total_edges}")
    print(f"Score: {total_edges / 611}")
