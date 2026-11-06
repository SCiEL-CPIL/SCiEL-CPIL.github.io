import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = start_time + 118
    
    n = 16
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_id = {t: i for i, t in enumerate(triples)}
    
    # Precompute K(2,2,2)s
    k222_triples = []
    triple_to_k222s = [[] for _ in range(num_triples)]
    
    def get_partitions(vertices):
        v = list(vertices)
        partitions = []
        for i in range(1, 6):
            pair1 = (v[0], v[i])
            remaining = [v[j] for j in range(1, 6) if j != i]
            for j in range(1, 4):
                pair2 = (remaining[0], remaining[j])
                if j == 1:
                    pair3 = (remaining[2], remaining[3])
                elif j == 2:
                    pair3 = (remaining[1], remaining[3])
                else:
                    pair3 = (remaining[1], remaining[2])
                partitions.append([pair1, pair2, pair3])
        return partitions

    for six_subset in combinations(range(n), 6):
        for partition in get_partitions(six_subset):
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
    
    class Solver:
        def __init__(self):
            self.present = set()
            self.k_counts = [0] * num_k222s
            
        def can_add(self, t_id):
            if t_id in self.present:
                return True
            for k_idx in triple_to_k222s[t_id]:
                if self.k_counts[k_idx] == 7:
                    return False
            return True
            
        def add(self, t_id):
            if not self.can_add(t_id):
                return False
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
    
    # Helper to polish the solution (add any missing triple that fits)
    def polish(solver):
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for t_id in range(num_triples):
                if t_id not in solver.present:
                    if solver.add(t_id):
                        improved = True
            if time.time() > DEADLINE:
                return

    # Helper to do a random valid 1-swap
    def random_swap(solver, rng):
        if not solver.present:
            return False
        
        present_list = list(solver.present)
        t_out = rng.choice(present_list)
        
        for _ in range(20):
            t_in = rng.randint(0, num_triples - 1)
            if t_in not in solver.present:
                break
        else:
            return False
            
        solver.remove(t_out)
        if solver.add(t_in):
            return True
        else:
            solver.add(t_out)
            return False

    # Helper to do a random valid 2-swap (remove 2, add 2)
    def random_2swap(solver, rng):
        if len(solver.present) < 2:
            return False
        
        present_list = list(solver.present)
        t_out1, t_out2 = rng.sample(present_list, 2)
        
        found_in1 = found_in2 = False
        for _ in range(30):
            if not found_in1:
                t_in1 = rng.randint(0, num_triples - 1)
                if t_in1 not in solver.present and t_in1 != t_out2:
                    found_in1 = True
            if not found_in2:
                t_in2 = rng.randint(0, num_triples - 1)
                if t_in2 not in solver.present and t_in2 != t_in1 and t_in2 != t_out1:
                    found_in2 = True
            if found_in1 and found_in2:
                break
        else:
            return False
            
        solver.remove(t_out1)
        solver.remove(t_out2)
        ok1 = solver.add(t_in1)
        ok2 = solver.add(t_in2)
        
        if ok1 and ok2:
            return True
        else:
            solver.remove(t_in1)
            solver.remove(t_in2)
            solver.add(t_out1)
            solver.add(t_out2)
            return False

    # Multiple restarts with different random seeds
    for seed in range(30):
        if time.time() > DEADLINE:
            break
        
        rng = random.Random(seed * 7919 + 1)
        solver = Solver()
        
        # Greedy Init with random order
        order = list(range(num_triples))
        rng.shuffle(order)
        for t_id in order:
            if time.time() > DEADLINE:
                break
            solver.add(t_id)
            
        # Polish
        polish(solver)
        
        if solver.size() > best_size:
            best_size = solver.size()
            best_set = set(solver.present)
        
        # Simulated Annealing / Random Walk
        # Run for remaining time, distributed across restarts
        sa_deadline = DEADLINE
        
        while time.time() < sa_deadline:
            # Perform a mix of 1-swaps and 2-swaps
            for _ in range(30):
                if time.time() > sa_deadline:
                    break
                if rng.random() < 0.6:
                    random_swap(solver, rng)
                else:
                    random_2swap(solver, rng)
            
            # Polish after each batch
            polish(solver)
            
            if solver.size() > best_size:
                best_size = solver.size()
                best_set = set(solver.present)
                
    return [list(triples[t_id]) for t_id in best_set]
