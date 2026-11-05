import sys
sys.setrecursionlimit(10000)

N = 18

# EVOLVE-BLOCK-START
def construct():
    def sort_network(n):
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        n_e = (n + 1) // 2
        n_o = n // 2
        
        # 1. Sort evens
        evens_comps = sort_network(n_e)
        mapped_evens = [(2*i, 2*j) for i, j in evens_comps]
        
        # 2. Sort odds
        odds_comps = sort_network(n_o)
        mapped_odds = [(2*i+1, 2*j+1) for i, j in odds_comps]
        
        # 3. Merge
        def merge(ne, no, offset):
            if ne == 0 or no == 0:
                return []
            
            comps = []
            
            # Compare first elements
            comps.append((offset, offset + 1))
            
            # Recursively merge
            # Sub-network on wires offset+1 ... offset+ne+no-1
            # "Evens" of sub are original odds (size no)
            # "Odds" of sub are original evens excluding first (size ne-1)
            comps.extend(merge(no, ne - 1, offset + 1))
            
            # Final compare if both have more than 1 element
            if ne > 1 and no > 1:
                last_e = offset + 2 * (ne - 1)
                last_o = offset + 1 + 2 * (no - 1)
                comps.append((last_e, last_o))
                
            return comps

        merge_comps = merge(n_e, n_o, 0)
        
        return mapped_evens + mapped_odds + merge_comps

    return sort_network(N)
# EVOLVE-BLOCK-END
