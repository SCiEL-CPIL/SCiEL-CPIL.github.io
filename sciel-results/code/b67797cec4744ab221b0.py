import time
import sys

# Increase recursion limit just in case, though depth is < 128
sys.setrecursionlimit(200)

class Solver:
    def __init__(self):
        self.best_len = 0
        self.best_seq = []
        self.deadline = time.time() + 50.0  # 50s safety margin
        self.timeout = False
        self.nodes_explored = 0

    def check_conflict(self, seq, products_set, x):
        """
        Check if adding x to seq creates any duplicate product.
        seq: list of ints
        products_set: set of all products of consecutive blocks in seq
        x: int to add
        Returns: (bool, set_of_new_products)
        """
        new_prods = set()
        prod = x
        new_prods.add(prod)
        if prod in products_set:
            return False, None
        
        # Multiply by elements of seq in reverse order
        for a in reversed(seq):
            prod *= a
            new_prods.add(prod)
            if prod in products_set:
                return False, None
        
        return True, new_prods

    def search(self, seq, products_set, depth):
        if self.timeout:
            return
        
        self.nodes_explored += 1
        if self.nodes_explored % 5000 == 0:
            if time.time() > self.deadline:
                self.timeout = True
                return

        current_len = len(seq)
        
        # Update best
        if current_len > self.best_len:
            self.best_len = current_len
            self.best_seq = seq.copy()
        
        # Pruning: Can we possibly beat the best?
        # Max possible extension is 128 - seq[-1] (if seq is not empty)
        if seq:
            max_possible = 128 - seq[-1]
        else:
            max_possible = 127  # 2 to 128
            
        if current_len + max_possible <= self.best_len:
            return

        # Find candidates
        start_val = seq[-1] + 1 if seq else 2
        end_val = 129
        
        # Heuristic: Try candidates that are "compatible" with more future numbers?
        # For now, iterate from start_val to end_val.
        # To find long sequences, smaller numbers are generally better to include early.
        
        for x in range(start_val, end_val):
            if self.timeout:
                return
            
            valid, new_prods = self.check_conflict(seq, products_set, x)
            if valid:
                # Recurse
                new_seq = seq + [x]
                new_products_set = products_set.union(new_prods)
                self.search(new_seq, new_products_set, depth + 1)
                
                # Backtrack
                # Note: new_prods is a set, so union is fine.
                # We don't need to explicitly remove from products_set because we pass a new set.
                # However, creating new sets is expensive.
                # Optimization: Use a global set and add/remove.
                # Given the small size, copying sets might be okay, but let's optimize if needed.
                # For now, let's stick to immutable sets for safety.

def construct():
    solver = Solver()
    
    # Initial greedy to set a good lower bound
    greedy_seq = []
    greedy_prods = set()
    for x in range(2, 129):
        valid, new_prods = solver.check_conflict(greedy_seq, greedy_prods, x)
        if valid:
            greedy_seq.append(x)
            greedy_prods.update(new_prods)
    
    if len(greedy_seq) > solver.best_len:
        solver.best_len = len(greedy_seq)
        solver.best_seq = greedy_seq.copy()

    # Start search from empty sequence
    solver.search([], set(), 0)
    
    return solver.best_seq
