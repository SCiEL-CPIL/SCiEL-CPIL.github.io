import time
import random
from collections import defaultdict
import itertools

def construct():
    deadline = time.time() + 115
    
    # Fallback: n=17 quadratic residue coloring
    def fallback():
        n = 17
        squares = {x * x % 17 for x in range(1, 17)}
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(0 if (j - i) % 17 in squares else 1)
        return {'n': n, 'colors': colors}
    
    best_result = fallback()
    
    # Precompute K2,2,2 structures for a given n
    # A K2,2,2 is a complete tripartite graph K_{2,2,2} with parts of size 2.
    # Vertices 0..5, parts {0,1}, {2,3}, {4,5}.
    # Edges are all pairs between different parts.
    # There are 15 ways to partition 6 vertices into 3 pairs.
    # For each subset of 6 vertices, and each partition into 3 pairs,
    # if all 12 edges between the pairs are monochromatic, we have a violation.
    
    # To be efficient, we represent the coloring as a numpy array or list.
    # We will try to find a valid coloring for n=29, then 28, etc.
    
    for n in [29, 28, 27]:
        if time.time() > deadline - 10:
            break
        
        # Generate all 6-sets and their partitions
        # This is expensive, so we do it once per n? No, it depends on n.
        # C(29, 6) = 501,870. Partitions = 15. Total checks per full scan = 7.5M.
        # This is too slow for Python in a loop if we do full scans.
        # We need incremental updates.
        
        # Structure:
        # 1. Map each edge (i,j) to its index.
        # 2. For each potential K2,2,2 instance (defined by 6 vertices and a partition),
        #    store the list of edge indices.
        # 3. Maintain a count for each instance: how many of its 12 edges are "bad" (non-monochromatic with the majority? or just count mismatches?)
        #    Actually, a violation is if ALL 12 edges are the SAME color.
        #    So for each instance, we can track:
        #    count_0 = number of edges colored 0
        #    count_1 = number of edges colored 1
        #    Violation if count_0 == 12 or count_1 == 12.
        
        edge_to_idx = {}
        edges = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_to_idx[(i, j)] = idx
                edges.append((i, j))
                idx += 1
        
        num_edges = idx
        
        # Generate all 6-sets and partitions
        # This step is heavy. Let's see if we can do it within the time budget.
        # C(29,6) * 15 = ~7.5 million instances.
        # Storing 7.5M lists of 12 ints is ~90M ints. Too much memory? 
        # 90M * 4 bytes (int32) = 360 MB. Might be risky but NumPy can handle it?
        # Or we can store them as a flat array and offsets.
        # Let's try to generate them on the fly for the initial random start, 
        # then update incrementally.
        # But we need the list of instances to know what to update when flipping an edge.
        # So we need a map: edge_idx -> list of instance_ids involving this edge.
        
        # Let's try to build this structure.
        instances = [] # List of lists of edge indices
        edge_to_instances = defaultdict(list) # edge_idx -> list of instance_ids
        
        start_gen = time.time()
        for vertices in itertools.combinations(range(n), 6):
            v0, v1, v2, v3, v4, v5 = vertices
            # Generate all 15 partitions of 6 items into 3 pairs
            # Partitions of {0,1,2,3,4,5} into 3 sets of size 2
            # 1. (0,1),(2,3),(4,5)
            # 2. (0,1),(2,4),(3,5)
            # ...
            # We can hardcode the 15 partitions or generate them.
            # Hardcoding is safer for speed.
            partitions = [
                [(0,1),(2,3),(4,5)],
                [(0,1),(2,4),(3,5)],
                [(0,1),(2,5),(3,4)],
                [(0,2),(1,3),(4,5)],
                [(0,2),(1,4),(3,5)],
                [(0,2),(1,5),(3,4)],
                [(0,3),(1,2),(4,5)],
                [(0,3),(1,4),(2,5)],
                [(0,3),(1,5),(2,4)],
                [(0,4),(1,2),(3,5)],
                [(0,4),(1,3),(2,5)],
                [(0,4),(1,5),(2,3)],
                [(0,5),(1,2),(3,4)],
                [(0,5),(1,3),(2,4)],
                [(0,5),(1,4),(2,3)]
            ]
            
            for part in partitions:
                edge_indices = []
                valid = True
                for pair in part:
                    p0, p1 = pair
                    # Edges between p0,p1 and other pairs
                    # Wait, K2,2,2 edges are ALL pairs between different parts.
                    # So for partition P1, P2, P3:
                    # Edges are all (u,v) where u in P_a, v in P_b, a!=b.
                    # There are 2*2=4 edges between P1 and P2.
                    # 4 edges between P1 and P3.
                    # 4 edges between P2 and P3.
                    # Total 12 edges.
                    pass
                
                # Construct edge indices for this specific partition of this specific 6-set
                # Part 0: vertices[part[0][0]], vertices[part[0][1]]
                # Part 1: vertices[part[1][0]], vertices[part[1][1]]
                # Part 2: vertices[part[2][0]], vertices[part[2][1]]
                
                p0_v = [vertices[part[0][0]], vertices[part[0][1]]]
                p1_v = [vertices[part[1][0]], vertices[part[1][1]]]
                p2_v = [vertices[part[2][0]], vertices[part[2][1]]]
                
                # Edges between p0 and p1
                for u in p0_v:
                    for v in p1_v:
                        if u < v:
                            edge_indices.append(edge_to_idx[(u, v)])
                        else:
                            edge_indices.append(edge_to_idx[(v, u)])
                # Edges between p0 and p2
                for u in p0_v:
                    for v in p2_v:
                        if u < v:
                            edge_indices.append(edge_to_idx[(u, v)])
                        else:
                            edge_indices.append(edge_to_idx[(v, u)])
                # Edges between p1 and p2
                for u in p1_v:
                    for v in p2_v:
                        if u < v:
                            edge_indices.append(edge_to_idx[(u, v)])
                        else:
                            edge_indices.append(edge_to_idx[(v, u)])
                
                instance_id = len(instances)
                instances.append(edge_indices)
                for e_idx in edge_indices:
                    edge_to_instances[e_idx].append(instance_id)
        
        if time.time() - start_gen > 20:
            continue # Precomputation too slow, skip this n
        
        # Initialize random coloring
        colors = [random.randint(0, 1) for _ in range(num_edges)]
        
        # Initialize counts for each instance
        # count_0[instance_id], count_1[instance_id]
        count_0 = [0] * len(instances)
        count_1 = [0] * len(instances)
        
        for inst_id, edge_indices in enumerate(instances):
            for e_idx in edge_indices:
                if colors[e_idx] == 0:
                    count_0[inst_id] += 1
                else:
                    count_1[inst_id] += 1
        
        # Count violations
        num_violations = sum(1 for i in range(len(instances)) if count_0[i] == 12 or count_1[i] == 12)
        
        if num_violations == 0:
            return {'n': n, 'colors': colors}
        
        # Local Search
        # While num_violations > 0 and time remains:
        #   Pick a random violating instance
        #   Pick a random edge in it
        #   Flip the edge
        #   Update counts for all instances involving this edge
        #   Update num_violations
        
        # To pick a random violating instance efficiently, we can maintain a list of violating instance IDs.
        # Or just pick a random instance and check if it's violating?
        # Given there are ~7.5M instances, picking randomly and checking is O(1) per check.
        # But if violations are rare, we might waste time.
        # Better: maintain a set/list of violating instances.
        
        violating_set = set()
        for i in range(len(instances)):
            if count_0[i] == 12 or count_1[i] == 12:
                violating_set.add(i)
        
        max_steps = 100000
        step = 0
        while num_violations > 0 and step < max_steps:
            if time.time() > deadline - 5:
                break
            
            if not violating_set:
                break
                
            # Pick random violating instance
            inst_id = random.choice(list(violating_set))
            edge_indices = instances[inst_id]
            
            # Pick random edge in this instance
            e_idx = random.choice(edge_indices)
            
            # Flip edge
            old_color = colors[e_idx]
            new_color = 1 - old_color
            colors[e_idx] = new_color
            
            # Update counts
            for inst_id_2 in edge_to_instances[e_idx]:
                if old_color == 0:
                    count_0[inst_id_2] -= 1
                    count_1[inst_id_2] += 1
                else:
                    count_0[inst_id_2] += 1
                    count_1[inst_id_2] -= 1
                
                was_violating = (count_0[inst_id_2] == 12) or (count_1[inst_id_2] == 12)
                # Wait, we need to check BEFORE update?
                # No, we updated the counts above. So check now.
                # But we need to know if it WAS violating to remove from set, and if it IS violating to add.
                
                # Let's redo this logic carefully.
                # We changed counts.
                # Check old state? No, we already changed counts.
                # We need to know if the state changed from non-violating to violating, or vice versa.
                
                # Let's save old counts before update?
                # Or just check if it's in violating_set and if it still is?
                
                # Better approach:
                # Before update: was_viol = (count_0==12 or count_1==12)
                # After update: is_viol = (count_0==12 or count_1==12)
                
                # Let's re-implement the update loop properly.
            
            # Re-implement update loop
            # We already flipped colors[e_idx].
            # We need to update counts and violating_set.
            
            # We'll do it again properly.
            # First, revert the flip? No, let's just do it right from the start of the step.
            
            # Let's restart the inner loop logic.
            # We picked e_idx.
            # Old color: old_color. New color: new_color.
            # colors[e_idx] is already new_color.
            
            for inst_id_2 in edge_to_instances[e_idx]:
                # Update counts
                if old_color == 0:
                    count_0[inst_id_2] -= 1
                    count_1[inst_id_2] += 1
                else:
                    count_0[inst_id_2] += 1
                    count_1[inst_id_2] -= 1
                
                is_viol = (count_0[inst_id_2] == 12) or (count_1[inst_id_2] == 12)
                
                if is_viol:
                    violating_set.add(inst_id_2)
                else:
                    violating_set.discard(inst_id_2)
            
            # Update num_violations
            num_violations = len(violating_set)
            step += 1
            
            if num_violations == 0:
                return {'n': n, 'colors': colors}
        
        if num_violations == 0:
            return {'n': n, 'colors': colors}
    
    return best_result

# Note: The above code has a bug in the update logic where I didn't correctly handle the transition.
# Specifically, `was_violating` was not checked correctly.
# Let's rewrite the update part more carefully in the final code.

def construct():
    deadline = time.time() + 115
    
    # Fallback: n=17 quadratic residue coloring
    def fallback():
        n = 17
        squares = {x * x % 17 for x in range(1, 17)}
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(0 if (j - i) % 17 in squares else 1)
        return {'n': n, 'colors': colors}
    
    best_result = fallback()
    
    for n in [29, 28, 27]:
        if time.time() > deadline - 10:
            break
        
        edge_to_idx = {}
        edges = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_to_idx[(i, j)] = idx
                edges.append((i, j))
                idx += 1
        
        num_edges = idx
        
        instances = []
        edge_to_instances = defaultdict(list)
        
        start_gen = time.time()
        # Generate 6-sets and partitions
        # Optimization: Only generate if we have enough time
        # C(29,6) is ~500k. 15 partitions each.
        # This loop might take a few seconds.
        
        for vertices in itertools.combinations(range(n), 6):
            v = vertices
            # Hardcoded partitions of indices 0..5 into 3 pairs
            # (a,b),(c,d),(e,f)
            partitions = [
                [(0,1),(2,3),(4,5)], [(0,1),(2,4),(3,5)], [(0,1),(2,5),(3,4)],
                [(0,2),(1,3),(4,5)], [(0,2),(1,4),(3,5)], [(0,2),(1,5),(3,4)],
                [(0,3),(1,2),(4,5)], [(0,3),(1,4),(2,5)], [(0,3),(1,5),(2,4)],
                [(0,4),(1,2),(3,5)], [(0,4),(1,3),(2,5)], [(0,4),(1,5),(2,3)],
                [(0,5),(1,2),(3,4)], [(0,5),(1,3),(2,4)], [(0,5),(1,4),(2,3)]
            ]
            
            for part in partitions:
                p0 = [v[part[0][0]], v[part[0][1]]]
                p1 = [v[part[1][0]], v[part[1][1]]]
                p2 = [v[part[2][0]], v[part[2][1]]]
                
                edge_indices = []
                # Edges between p0, p1
                for u in p0:
                    for w in p1:
                        edge_indices.append(edge_to_idx[(min(u,w), max(u,w))])
                # Edges between p0, p2
                for u in p0:
                    for w in p2:
                        edge_indices.append(edge_to_idx[(min(u,w), max(u,w))])
                # Edges between p1, p2
                for u in p1:
                    for w in p2:
                        edge_indices.append(edge_to_idx[(min(u,w), max(u,w))])
                
                inst_id = len(instances)
                instances.append(edge_indices)
                for e in edge_indices:
                    edge_to_instances[e].append(inst_id)
        
        if time.time() - start_gen > 20:
            continue
        
        colors = [random.randint(0, 1) for _ in range(num_edges)]
        
        count_0 = [0] * len(instances)
        count_1 = [0] * len(instances)
        
        for i in range(len(instances)):
            for e in instances[i]:
                if colors[e] == 0:
                    count_0[i] += 1
                else:
                    count_1[i] += 1
        
        violating_set = set()
        for i in range(len(instances)):
            if count_0[i] == 12 or count_1[i] == 12:
                violating_set.add(i)
        
        num_violations = len(violating_set)
        if num_violations == 0:
            return {'n': n, 'colors': colors}
        
        step = 0
        while num_violations > 0 and step < 100000:
            if time.time() > deadline - 5:
                break
            
            if not violating_set:
                break
            
            inst_id = random.choice(list(violating_set))
            edge_indices = instances[inst_id]
            e_idx = random.choice(edge_indices)
            
            old_color = colors[e_idx]
            new_color = 1 - old_color
            colors[e_idx] = new_color
            
            for inst_id_2 in edge_to_instances[e_idx]:
                # Calculate violation status before update?
                # We need to know if it was violating and if it is violating now.
                was_viol = (count_0[inst_id_2] == 12) or (count_1[inst_id_2] == 12)
                
                if old_color == 0:
                    count_0[inst_id_2] -= 1
                    count_1[inst_id_2] += 1
                else:
                    count_0[inst_id_2] += 1
                    count_1[inst_id_2] -= 1
                
                is_viol = (count_0[inst_id_2] == 12) or (count_1[inst_id_2] == 12)
                
                if was_viol and not is_viol:
                    violating_set.discard(inst_id_2)
                elif not was_viol and is_viol:
                    violating_set.add(inst_id_2)
            
            num_violations = len(violating_set)
            step += 1
            
            if num_violations == 0:
                return {'n': n, 'colors': colors}
    
    return best_result
