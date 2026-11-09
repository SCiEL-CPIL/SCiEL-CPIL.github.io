import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)  # 560
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute all K(2,2,2) forbidden configurations
    # A K(2,2,2) is defined by 6 distinct points {a,a',b,b',c,c'}
    # The 8 transversal triples are all combinations choosing one from each pair
    # We need to find which triples participate in which K(2,2,2)s
    
    # For each K(2,2,2) configuration, store the 8 triples that form it
    # Two triples conflict if they both appear in some K(2,2,2)
    # Actually, the constraint is: no K(2,2,2) should have ALL 8 of its transversals present
    
    # Strategy: precompute for each triple, the list of K(2,2,2)s it belongs to
    # Then when checking if adding triple t would create a forbidden config,
    # we check if any K(2,2,2) containing t has all other 7 transversals present
    
    # First, enumerate all K(2,2,2) structures
    # A K(2,2,2) is determined by partitioning 6 points into 3 pairs
    # For 6 points, there are 15 ways to partition into 3 pairs
    # But we only care about the 8 transversals (one from each pair)
    
    # Let's precompute: for each triple, which 6-point sets and pairings could form a K(2,2,2) with it
    # Actually, let's precompute all forbidden 8-sets (the 8 transversals of each K(2,2,2))
    
    # There are C(16,6) = 8008 ways to choose 6 points
    # For each 6-point set, there are 15 pairings into 3 pairs
    # For each pairing, there are 8 transversals
    # Total K(2,2,2) instances: 8008 * 15 = 120120
    
    # For each such instance, store the 8 triples
    # Then for each triple, store which instances it belongs to
    
    triple_to_instances = defaultdict(list)
    
    for six_points in itertools.combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = six_points
        # All ways to partition 6 points into 3 pairs
        # Fix p0's partner, then partition remaining 4
        pairings = [
            ((p0,p1),(p2,p3),(p4,p5)),
            ((p0,p1),(p2,p4),(p3,p5)),
            ((p0,p1),(p2,p5),(p3,p4)),
            ((p0,p2),(p1,p3),(p4,p5)),
            ((p0,p2),(p1,p4),(p3,p5)),
            ((p0,p2),(p1,p5),(p3,p4)),
            ((p0,p3),(p1,p2),(p4,p5)),
            ((p0,p3),(p1,p4),(p2,p5)),
            ((p0,p3),(p1,p5),(p2,p4)),
            ((p0,p4),(p1,p2),(p3,p5)),
            ((p0,p4),(p1,p3),(p2,p5)),
            ((p0,p4),(p1,p5),(p2,p3)),
            ((p0,p5),(p1,p2),(p3,p4)),
            ((p0,p5),(p1,p3),(p2,p4)),
            ((p0,p5),(p1,p4),(p2,p3)),
        ]
        
        for pairs in pairings:
            (a, a_prime), (b, b_prime), (c, c_prime) = pairs
            # 8 transversals
            transversals = []
            for x in (a, a_prime):
                for y in (b, b_prime):
                    for z in (c, c_prime):
                        t = tuple(sorted((x, y, z)))
                        transversals.append(t)
            
            instance_id = len(triple_to_instances)  # Use as identifier
            for t in transversals:
                triple_to_instances[t].append(instance_id)
    
    # Now for each triple, we have a list of instance IDs it belongs to
    # For fast checking: maintain a count for each instance of how many of its 8 triples are present
    # When adding triple t, check if any instance containing t would reach count 8
    
    # Precompute for each instance, the set of 8 triples
    # We need to map instance_id -> list of 8 triples
    # But we didn't store that directly. Let's rebuild.
    
    # Actually, let's redo this more carefully
    all_instances = []  # Each is a frozenset of 8 triples
    triple_to_instances = defaultdict(list)
    
    for six_points in itertools.combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = six_points
        pairings = [
            ((p0,p1),(p2,p3),(p4,p5)),
            ((p0,p1),(p2,p4),(p3,p5)),
            ((p0,p1),(p2,p5),(p3,p4)),
            ((p0,p2),(p1,p3),(p4,p5)),
            ((p0,p2),(p1,p4),(p3,p5)),
            ((p0,p2),(p1,p5),(p3,p4)),
            ((p0,p3),(p1,p2),(p4,p5)),
            ((p0,p3),(p1,p4),(p2,p5)),
            ((p0,p3),(p1,p5),(p2,p4)),
            ((p0,p4),(p1,p2),(p3,p5)),
            ((p0,p4),(p1,p3),(p2,p5)),
            ((p0,p4),(p1,p5),(p2,p3)),
            ((p0,p5),(p1,p2),(p3,p4)),
            ((p0,p5),(p1,p3),(p2,p4)),
            ((p0,p5),(p1,p4),(p2,p3)),
        ]
        
        for pairs in pairings:
            (a, a_prime), (b, b_prime), (c, c_prime) = pairs
            transversals = []
            for x in (a, a_prime):
                for y in (b, b_prime):
                    for z in (c, c_prime):
                        t = tuple(sorted((x, y, z)))
                        transversals.append(t)
            
            instance_id = len(all_instances)
            all_instances.append(transversals)
            for t in transversals:
                triple_to_instances[t].append(instance_id)
    
    num_instances = len(all_instances)
    
    def check_valid(triple, present_set, instance_counts):
        """Check if adding triple would create a forbidden K(2,2,2)"""
        for inst_id in triple_to_instances[triple]:
            if instance_counts[inst_id] >= 7:
                # This instance already has 7 of its 8 triples present
                # Adding this triple would make it 8 -> forbidden
                return False
        return True
    
    def add_triple(triple, present_set, instance_counts):
        """Add triple and update counts"""
        present_set.add(triple)
        for inst_id in triple_to_instances[triple]:
            instance_counts[inst_id] += 1
    
    def remove_triple(triple, present_set, instance_counts):
        """Remove triple and update counts"""
        present_set.remove(triple)
        for inst_id in triple_to_instances[triple]:
            instance_counts[inst_id] -= 1
    
    best_size = 0
    best_set = set()
    
    # Greedy + local search with multiple restarts
    for trial in range(200):
        if time.time() > deadline - 15.0:
            break
        
        random.seed(trial * 7919 + 13)
        
        # Greedy phase: add triples in random order
        order = list(range(num_triples))
        random.shuffle(order)
        
        present_set = set()
        instance_counts = [0] * num_instances
        
        for idx in order:
            if time.time() > deadline - 10.0:
                break
            t = all_triples[idx]
            if check_valid(t, present_set, instance_counts):
                add_triple(t, present_set, instance_counts)
        
        # Local search: try to add any missing triple
        improved = True
        while improved:
            if time.time() > deadline - 10.0:
                break
            improved = False
            for idx in range(num_triples):
                t = all_triples[idx]
                if t in present_set:
                    continue
                if check_valid(t, present_set, instance_counts):
                    add_triple(t, present_set, instance_counts)
                    improved = True
                    break
        
        # 2-swap local search: remove one, add two
        swap_improved = True
        while swap_improved:
            if time.time() > deadline - 10.0:
                break
            swap_improved = False
            
            # Try removing each present triple and see if we can add two
            for t_remove in list(present_set):
                if time.time() > deadline - 10.0:
                    break
                
                # Temporarily remove
                remove_triple(t_remove, present_set, instance_counts)
                
                # Try to add two triples
                added = []
                for idx in range(num_triples):
                    if time.time() > deadline - 10.0:
                        break
                    t_add = all_triples[idx]
                    if t_add in present_set or t_add == t_remove:
                        continue
                    if check_valid(t_add, present_set, instance_counts):
                        add_triple(t_add, present_set, instance_counts)
                        added.append(t_add)
                        if len(added) >= 2:
                            break
                
                if len(added) >= 2:
                    swap_improved = True
                else:
                    # Put back what we added and restore t_remove
                    for t in added:
                        remove_triple(t, present_set, instance_counts)
                    add_triple(t_remove, present_set, instance_counts)
        
        if len(present_set) > best_size:
            best_size = len(present_set)
            best_set = set(present_set)
    
    result = [list(t) for t in best_set]
    return result
