import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110
    
    n = 16
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_id = {t: i for i, t in enumerate(triples)}
    
    # Precompute K(2,2,2)s
    # A K(2,2,2) is defined by 6 distinct vertices partitioned into 3 pairs.
    # We iterate over all 6-subsets.
    # For each 6-subset, there are 15 ways to partition into 3 pairs.
    # We generate the 8 transversals for each partition.
    
    # Mapping: k222_index -> list of 8 triple_ids
    k222_triples = []
    # Mapping: triple_id -> list of k222_indices
    triple_to_k222s = [[] for _ in range(num_triples)]
    
    def get_partitions(vertices):
        # vertices is a list of 6 distinct ints
        # Returns list of 15 partitions, each a list of 3 pairs (tuples)
        v = list(vertices)
        # Fix v[0] with one of the other 5
        # Then partition the remaining 4 into 2 pairs
        partitions = []
        for i in range(1, 6):
            pair1 = (v[0], v[i])
            remaining = [v[j] for j in range(1, 6) if j != i]
            # remaining has 4 elements
            # Fix remaining[0] with one of the other 3
            for j in range(1, 4):
                pair2 = (remaining[0], remaining[j])
                pair3 = (remaining[1], remaining[2]) if remaining[j] == remaining[1] else (remaining[2], remaining[3])
                # Wait, need to be careful with indices
                # remaining = [r0, r1, r2, r3]
                # pair2 = (r0, rj) where j in 1,2,3
                # pair3 is the other two
                if j == 1:
                    pair3 = (remaining[2], remaining[3])
                elif j == 2:
                    pair3 = (remaining[1], remaining[3])
                else: # j == 3
                    pair3 = (remaining[1], remaining[2])
                
                partitions.append([pair1, pair2, pair3])
        return partitions

    # Generate all K(2,2,2)s
    for six_subset in combinations(range(n), 6):
        for partition in get_partitions(six_subset):
            # partition is [ (a, a'), (b, b'), (c, c') ]
            # The 8 transversals are (x, y, z) where x in P1, y in P2, z in P3
            t_ids = []
            for x in partition[0]:
                for y in partition[1]:
                    for z in partition[2]:
                        t = tuple(sorted((x, y, z)))
                        t_id = triple_to_id[t]
                        t_ids.append(t_id)
            
            k_idx = len(k222_triples)
            k222_triples.append(t_ids)
            
            for t_id in t_ids:
                triple_to_k222s[t_id].append(k_idx)
                
    num_k222s = len(k222_triples)
    
    # Solver Class
    class Solver:
        def __init__(self):
            self.present = set()
            self.k_counts = [0] * num_k222s
            
        def add(self, t_id):
            if t_id in self.present:
                return True # Already present
            # Check if safe
            for k_idx in triple_to_k222s[t_id]:
                if self.k_counts[k_idx] == 7:
                    return False
            # Add
            self.present.add(t_id)
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] += 1
            return True
            
        def remove(self, t_id):
            if t_id not in self.present:
                return
            self.present.remove(t_id)
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] -= 1
                
        def size(self):
            return len(self.present)

    best_size = 0
    best_set = set()
    
    rng = random.Random(42)
    
    # Run multiple restarts
    while time.time() < DEADLINE:
        solver = Solver()
        
        # Greedy Phase
        order = list(range(num_triples))
        rng.shuffle(order)
        
        for t_id in order:
            if time.time() > DEADLINE:
                break
            solver.add(t_id)
            
        # Local Search / Polish Phase
        # Try to add any missing triple
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            # We can iterate over all triples, or just missing ones
            # Iterating over all is fine (560)
            for t_id in range(num_triples):
                if t_id not in solver.present:
                    if solver.add(t_id):
                        improved = True
                        # If we added one, we might need to restart the loop to check others?
                        # Or just continue. Since adding one might block others, 
                        # but we only add if safe, so it won't break existing ones.
                        # However, it might make *other* missing triples unsafe? 
                        # No, adding a triple only increases counts. 
                        # It can only make *new* additions unsafe, not existing ones.
                        # So we can just continue.
                        
        # If stuck, try a kick
        if time.time() < DEADLINE:
            # Try to improve by removing one and adding more
            # This is a bit risky, let's just do more restarts if time permits
            # Or a simple 1-step local search:
            # Pick a random present triple, remove it.
            # Try to add any missing.
            # If size > best_size, update best.
            # If size < current_size, revert.
            
            # Let's try a few kicks
            for _ in range(10):
                if time.time() > DEADLINE:
                    break
                if not solver.present:
                    break
                # Pick random triple to remove
                t_remove = rng.choice(list(solver.present))
                solver.remove(t_remove)
                
                # Try to add missing
                local_improved = True
                while local_improved and time.time() < DEADLINE:
                    local_improved = False
                    for t_id in range(num_triples):
                        if t_id not in solver.present:
                            if solver.add(t_id):
                                local_improved = True
                                
                if solver.size() > best_size:
                    best_size = solver.size()
                    best_set = set(solver.present)
                    
                # Revert the removal if it didn't help immediately?
                # Or keep the state?
                # If we keep the state, we are at size-1 + added.
                # If added < 1, we are worse.
                # Let's revert to the state before the kick if we didn't improve global best?
                # Actually, just reverting the kick (re-adding t_remove) is safe.
                # But we might have added other things.
                # Simplest: If we didn't beat best_size, revert to the state before the kick.
                # To do that, we need to save the state.
                # Saving counts is expensive.
                # Let's just do restarts. The greedy + polish is likely strong enough.
                
        if solver.size() > best_size:
            best_size = solver.size()
            best_set = set(solver.present)
            
    return [list(triples[t_id]) for t_id in best_set]
