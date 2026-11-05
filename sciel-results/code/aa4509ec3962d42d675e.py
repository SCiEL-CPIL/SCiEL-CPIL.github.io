import time

def construct():
        start_time = time.time()
        deadline = start_time + 55  # Safety margin for 60s limit
        
        # Beam width
        W = 5000
        
        # Initial state: empty sequence, empty set of products
        # Using frozenset for products to make them hashable and immutable
        beam = [((), frozenset())]
        
        best_seq = []
        best_len = 0
        
        for x in range(2, 129):
            if time.time() > deadline:
                break
                
            next_beam = []
            
            for seq, prods in beam:
                # Option 1: Skip x
                next_beam.append((seq, prods))
                
                # Option 2: Include x
                # Check if strictly increasing
                if not seq or seq[-1] < x:
                    # Calculate new suffix products
                    # The new products are x, x*last, x*last*second_last, etc.
                    new_prods_list = []
                    prod = x
                    new_prods_list.append(prod)
                    
                    # Multiply by elements in reverse order
                    for a in reversed(seq):
                        prod *= a
                        new_prods_list.append(prod)
                    
                    # Check for collisions with existing products
                    # prods is a frozenset, so 'in' is O(1)
                    is_valid = True
                    for p in new_prods_list:
                        if p in prods:
                            is_valid = False
                            break
                    
                    if is_valid:
                        # Create new product set
                        # Union of frozenset and list
                        new_prods_set = prods.union(new_prods_list)
                        next_beam.append((seq + (x,), frozenset(new_prods_set)))
            
            # Prune the beam
            # 1. Deduplicate by sequence
            #    Since prods are determined by seq, we can just keep one instance per seq
            unique_states = {}
            for seq, prods in next_beam:
                if seq not in unique_states:
                    unique_states[seq] = prods
            
            # 2. Sort and keep top W
            #    Sort by:
            #    - Length (descending)
            #    - Sum of elements (ascending) - smaller sum implies smaller numbers, more room
            #    - Last element (ascending) - smaller last element allows more candidates
            
            sorted_seqs = sorted(
                unique_states.keys(), 
                key=lambda s: (-len(s), sum(s), s[-1] if s else 0)
            )
            
            beam = [(s, unique_states[s]) for s in sorted_seqs[:W]]
            
            # Update best solution found so far
            if sorted_seqs:
                top_seq = sorted_seqs[0]
                if len(top_seq) > best_len:
                    best_len = len(top_seq)
                    best_seq = list(top_seq)
        
        return best_seq
