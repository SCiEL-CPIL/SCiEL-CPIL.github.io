import sys
sys.setrecursionlimit(10000)

def construct():
    n = 256
    best_set = []
    best_size = 0
    
    # Initial greedy solution to set a good lower bound
    greedy_set = []
    greedy_pair_sums = set()
    for x in range(n, 0, -1):
        can_add = True
        # Check if x divides any pair sum in greedy_set
        # We need to check if any s in greedy_pair_sums is divisible by x
        # Note: greedy_pair_sums contains sums of pairs from greedy_set
        # If x divides s, then x | (b+c).
        # Wait, the condition is a | (b+c). Here a=x.
        # So if any pair sum in current set is a multiple of x, we cannot add x.
        
        # Optimization: Instead of storing all sums, we can just check.
        # But storing sums allows O(1) check for "is there a multiple of x?"
        # Actually, checking if any element in a set is divisible by x is O(|set|).
        # |set| can be up to ~10000. 256 * 10000 = 2.5M ops. This is fast.
        
        # Let's refine the greedy to be faster or just use it as seed.
        pass
        
    # Let's just run the B&B directly.
    # To make it efficient, we will use a class to hold state.
    
    class Solver:
        def __init__(self):
            self.n = n
            self.best_set = []
            self.best_size = 0
            self.node_count = 0
            self.max_nodes = 5000000 # Safety limit
            
        def can_add(self, x, current_set, pair_sums):
            # Check if x divides any sum in pair_sums
            # pair_sums contains b+c for all b,c in current_set, b<c
            # We need to check if any s in pair_sums is such that s % x == 0
            for s in pair_sums:
                if s % x == 0:
                    return False
            return True
            
        def solve(self, x, current_set, pair_sums):
            nonlocal best_set, best_size
            if self.node_count > self.max_nodes:
                return
                
            self.node_count += 1
            
            # Pruning: Upper bound on size
            # Max possible additional elements is x (since we go down to 1)
            # Actually, we can add at most x more elements (1..x)
            if len(current_set) + x <= self.best_size:
                return
                
            # Base case
            if x == 0:
                if len(current_set) > self.best_size:
                    self.best_size = len(current_set)
                    self.best_set = list(current_set)
                return
                
            # Option 1: Add x
            # Check if x can be added
            if self.can_add(x, current_set, pair_sums):
                # New pair sums: x + y for all y in current_set
                new_pair_sums = set(pair_sums)
                for y in current_set:
                    new_pair_sums.add(x + y)
                    
                current_set.append(x)
                self.solve(x - 1, current_set, new_pair_sums)
                current_set.pop()
                
            # Option 2: Skip x
            self.solve(x - 1, current_set, pair_sums)

    solver = Solver()
    solver.solve(n, [], set())
    
    return solver.best_set
