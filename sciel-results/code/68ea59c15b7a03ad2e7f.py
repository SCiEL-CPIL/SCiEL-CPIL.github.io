import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # Beam search
    # State: (sequence_tuple, product_set)
    # Initialize beam with single elements
    beam = []
    for x in range(2, 129):
        beam.append(((x,), {x}))
    
    best_sequence = []
    
    # We will expand until no more extensions are possible or time runs out
    # Max possible length is 127, but realistically ~20
    width = 100  # Beam width
    
    while beam and time.time() < deadline:
        next_beam = []
        # Sort current beam by last element to prioritize those with more room?
        # Actually, we should expand all and then prune.
        
        for seq, prods in beam:
            last = seq[-1]
            # Try all x > last
            for x in range(last + 1, 129):
                # Calculate new products
                # New products are x * (product of suffix of seq)
                # Suffix products: 1, seq[-1], seq[-1]*seq[-2], ...
                # We can compute these on the fly.
                
                # Optimization: 
                # We need to check if any new product is in prods.
                # New products:
                # p_0 = x
                # p_1 = x * seq[-1]
                # p_2 = x * seq[-1] * seq[-2]
                # ...
                
                # To speed this up, we can precompute suffix products of seq?
                # Or just compute incrementally.
                
                valid = True
                new_prods = set()
                current_prod = 1
                # Iterate backwards through seq
                for i in range(len(seq) - 1, -1, -1):
                    current_prod *= seq[i]
                    p = x * current_prod
                    if p in prods or p in new_prods:
                        valid = False
                        break
                    new_prods.add(p)
                
                if valid:
                    new_seq = seq + (x,)
                    new_prods.update(prods)
                    next_beam.append((new_seq, new_prods))
        
        if not next_beam:
            break
            
        # Prune next_beam to width
        # Heuristic: Keep states with smallest last element to allow more future choices
        next_beam.sort(key=lambda s: s[0][-1])
        beam = next_beam[:width]
        
        # Track best
        if len(beam[0][0]) > len(best_sequence):
            best_sequence = list(beam[0][0])
            
    # If beam is empty or time ran out, return best found
    if not best_sequence:
         # Fallback to greedy if beam search failed immediately (shouldn't happen)
         values, products = [], set()
         for x in range(2, 129):
             tails, prod = [x], x
             for a in reversed(values):
                 prod *= a
                 tails.append(prod)
             if not products.intersection(tails):
                 values.append(x)
                 products.update(tails)
         return values

    return best_sequence

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
