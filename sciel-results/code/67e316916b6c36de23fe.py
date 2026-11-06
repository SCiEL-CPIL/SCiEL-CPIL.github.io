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
    
    rng = random.Random(42)
    
    # Helper to polish the solution (add any missing triple that fits)
    def polish(solver):
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            # Iterate over missing triples. 
            # To be faster, we can iterate over all and check if missing.
            # Or maintain a list of missing.
            # For now, simple loop.
            for t_id in range(num_triples):
                if t_id not in solver.present:
                    if solver.add(t_id):
                        improved = True
                        if time.time() > DEADLINE:
                            return

    # Helper to do a random valid swap
    def random_swap(solver):
        if not solver.present:
            return False
        
        present_list = list(solver.present)
        t_out = rng.choice(present_list)
        
        # Find a random missing triple
        # We can sample random t_ids until we find a missing one
        # Or build a list of missing ones (expensive if done often)
        # Let's just try a few random missing triples
        # Since density is likely not extremely low, random sampling is okay.
        
        # To be efficient, let's pick a random t_id from 0..num_triples-1
        # and check if it's missing.
        
        # Actually, if we want to ensure we find a swap, we might need to try a few.
        # But for SA, we just pick one. If it's present, we skip or pick another.
        
        # Let's pick a random t_in from the missing ones.
        # To avoid O(N) scan every time, we can maintain a set of missing? 
        # Or just accept that we might pick a present one and retry a few times.
        
        attempts = 0
        while attempts < 10:
            t_in = rng.randint(0, num_triples - 1)
            if t_in not in solver.present:
                break
            attempts += 1
            
        if t_in in solver.present:
            return False
            
        # Try to swap: remove t_out, add t_in
        solver.remove(t_out)
        if solver.add(t_in):
            return True
        else:
            # Revert
            solver.add(t_out)
            return False

    while time.time() < DEADLINE:
        solver = Solver()
        
        # Greedy Init
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
        # We will perform a series of swaps to escape local optima
        # then polish again.
        
        # Parameters
        temp = 1.0
        cooling = 0.99
        
        # Run for a fixed number of steps or time
        # Let's run for ~2 seconds per restart? No, we have 110s total.
        # Let's run until time is up, but check best frequently.
        
        # Actually, let's do a fixed number of "epochs" of swaps + polish.
        # Or just a loop.
        
        while time.time() < DEADLINE:
            # Perform a batch of swaps
            # The number of swaps can be adaptive.
            # Let's do 50 swaps.
            for _ in range(50):
                if time.time() > DEADLINE:
                    break
                if random_swap(solver):
                    # Swap successful, state changed
                    pass
            
            # Polish
            polish(solver)
            
            if solver.size() > best_size:
                best_size = solver.size()
                best_set = set(solver.present)
                
            # If we haven't improved in a long time, restart?
            # For now, just keep going.
            
    return [list(triples[t_id]) for t_id in best_set]
