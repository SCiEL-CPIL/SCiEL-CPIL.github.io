import time
import random
import numpy as np

N = 512

def construct():
    deadline = time.time() + 58.0  # Safety margin for 60s limit
    
    # 1. Precompute 4-AP constraints
    # For each index i (0..N-1), we want to know which other indices j, k, l 
    # form a 4-AP with i.
    # A 4-AP is a, a+d, a+2d, a+3d.
    # If we have a set S, adding x creates a 4-AP if there exist y, z, w in S 
    # such that {x, y, z, w} is a 4-AP.
    
    # Let's precompute for each x, the list of "forbidden triples" (y, z, w).
    # If y, z, w are in S, x cannot be added.
    # This is the same logic as the current program, but we will use it differently.
    
    # Actually, for a swap (remove r, add a), we need to check:
    # 1. Is the set S \ {r} + {a} valid?
    #    This means:
    #    - No 4-AP is formed entirely within S \ {r} + {a}.
    #    - Since S was valid, the only new 4-APs must involve 'a' and at least 3 elements from S \ {r}.
    #    - Also, removing 'r' might "break" 4-APs, but since S was valid, there were no 4-APs to begin with.
    #    - So we just need to ensure that 'a' does not form a 4-AP with any triple from S \ {r}.
    
    # Let's precompute for each x, the set of triples T_x = {(y,z,w) | {x,y,z,w} is a 4-AP}.
    # To check if 'a' can be added to S' (where S' = S \ {r}), we check if any triple in T_a is a subset of S'.
    
    # For speed, we can represent S as a boolean array or bitset.
    # Using NumPy, we can check subset relations efficiently?
    # Maybe not directly for triples.
    
    # Alternative approach:
    # Precompute for each pair (r, a), the condition for swap validity.
    # This is too many pairs (512^2).
    
    # Let's stick to the "can_add" check but vectorize the "triple in set" check.
    # For a given 'a', we have a list of triples.
    # We can represent each triple as a bitmask (3 bits).
    # We can represent S as a bitset (Python int or NumPy array of 64-bit chunks).
    # Checking if a triple is in S: (S_mask & triple_mask) == triple_mask.
    
    # Let's use Python integers for the bitset as they are very fast for bit ops in Python 3.
    # But the current program already does this. The bottleneck is the loop over triples.
    
    # How many triples per x?
    # x can be a, a+d, a+2d, a+3d.
    # Max d is ~170.
    # For each d, x is in at most 4 APs? No, for a fixed x and d, x can be part of at most 4 APs?
    # Actually, for a fixed x, the number of 4-APs containing x is O(N).
    # Total triples for all x is O(N^2).
    # For N=512, this is manageable.
    
    forbidden = [[] for _ in range(N)]
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # 0-indexed
            i, j, k, l = a-1, b-1, c-1, e-1
            
            forbidden[i].append((j, k, l))
            forbidden[j].append((i, k, l))
            forbidden[k].append((i, j, l))
            forbidden[l].append((i, j, k))
            
    # Convert to lists of masks for faster intersection
    forbidden_masks = []
    for i in range(N):
        masks = []
        for t in forbidden[i]:
            m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            masks.append(m)
        forbidden_masks.append(masks)

    def can_add(bits, x):
        # Check if any forbidden mask for x is a subset of bits
        for m in forbidden_masks[x]:
            if (bits & m) == m:
                return False
        return True

    # 2. Initialization: Ternary Digit Heuristic
    # Numbers in [1, N] whose base-3 representation contains no 0s.
    # This set is known to be 3-AP free, and thus 4-AP free.
    init_set = set()
    for num in range(1, N + 1):
        x = num
        valid = True
        while x > 0:
            if x % 3 == 0:
                valid = False
                break
            x //= 3
        if valid:
            init_set.add(num - 1) # 0-indexed
            
    # Convert to bitset
    bits = 0
    for i in init_set:
        bits |= (1 << i)
    
    current_size = len(init_set)
    best_bits = bits
    best_size = current_size
    
    # 3. Local Search: Swap Optimization
    # We try to swap r (in S) with a (not in S).
    # Validity condition:
    # Let S' = S \ {r} \cup {a}.
    # S' is valid iff 'a' does not form a 4-AP with any triple in S \ {r}.
    # Let T_a be the set of triples for 'a'.
    # We need: For all t in T_a, t is NOT a subset of S \ {r}.
    # t is a subset of S \ {r} iff (t subset of S) AND (r not in t).
    # So the condition is: For all t in T_a, if (t subset of S), then r MUST be in t.
    # Equivalently: There is NO t in T_a such that (t subset of S) AND (r not in t).
    
    # Precompute for each 'a', the list of triples.
    # For a given state S (bits), and a candidate 'a' not in S:
    # We want to find if there exists a 'r' in S such that the swap is valid.
    # Actually, we want to maximize size. A swap keeps size constant.
    # To increase size, we need to add 'a' without removing 'r'? No, S is maximal?
    # If S is not maximal, we can just add 'a'.
    # First, try to add any 'a' not in S.
    
    # Phase 3a: Greedy Add
    # Scan all a not in S, if can_add(bits, a), add it.
    # Repeat until no more can be added.
    
    def greedy_add(bits):
        size = bin(bits).count('1')
        changed = True
        while changed:
            changed = False
            for a in range(N):
                if not (bits & (1 << a)):
                    if can_add(bits, a):
                        bits |= (1 << a)
                        size += 1
                        changed = True
        return bits, size
    
    bits, current_size = greedy_add(bits)
    best_bits = bits
    best_size = current_size
    
    # Phase 3b: Swap Search
    # We look for a swap (r, a) that maintains validity.
    # This allows us to "move" in the space of maximal sets.
    # Once we find a valid swap, we apply it.
    # We might also try "2-swap" or "kick" if stuck, but let's start with 1-swap.
    
    # To speed up swap check:
    # For a candidate 'a' not in S, let BadTriples_a = { t in T_a | t subset of S }.
    # A swap (r, a) is valid iff r is in ALL t in BadTriples_a.
    # i.e., r is in the intersection of all triples in BadTriples_a.
    # If BadTriples_a is empty, any r works (actually, if BadTriples_a is empty, 
    # then 'a' can be added directly, which should have been caught in greedy_add. 
    # Unless greedy_add missed it? No, greedy_add checks can_add. 
    # can_add(bits, a) is False iff there exists t in T_a such that t subset of bits.
    # So if can_add is False, BadTriples_a is not empty.)
    
    # So, for each a not in S where can_add(bits, a) is False:
    # 1. Find all t in T_a such that t subset of S.
    # 2. Compute intersection of indices in these triples.
    # 3. If intersection is non-empty, any r in intersection is a valid removal candidate.
    # 4. We pick one such r and perform the swap.
    
    # We repeat this until no valid swap is found.
    # Then we try a random "kick" (remove 2, add 1 or something) if stuck?
    # For now, just 1-swap local search.
    
    # To make it explore, we can randomize the order of 'a' and 'r'.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle candidates to avoid bias
        candidates = [a for a in range(N) if not (bits & (1 << a))]
        random.shuffle(candidates)
        
        for a in candidates:
            if time.time() > deadline:
                break
                
            if bits & (1 << a):
                continue
                
            # Check if a can be added (should be False if we are in swap phase, 
            # but maybe we missed some? No, greedy_add ran. But after swaps, 
            # new elements might become addable? 
            # Actually, if we swap r->a, the set changes. 
            # Let's just check validity of swap directly.)
            
            # Find bad triples for a
            bad_triples = []
            for t in forbidden[a]:
                m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
                if (bits & m) == m:
                    bad_triples.append(t)
            
            if not bad_triples:
                # This means a can be added!
                # This shouldn't happen if greedy_add was exhaustive, 
                # but let's handle it.
                bits |= (1 << a)
                current_size += 1
                improved = True
                # Restart greedy add?
                # For efficiency, just break and let the outer loop handle it?
                # Or just continue.
                continue
            
            # Find intersection of indices in bad_triples
            # Intersection of sets of indices
            common = set(bad_triples[0])
            for t in bad_triples[1:]:
                common &= set(t)
            
            if common:
                # We can remove any r in common
                # Pick one randomly or deterministically
                # To explore, pick randomly
                r = random.choice(list(common))
                
                # Perform swap
                bits &= ~(1 << r)
                bits |= (1 << a)
                # Size remains same
                improved = True
                # After a swap, the set has changed. 
                # We should probably restart the candidate scan or at least 
                # re-evaluate if new additions are possible.
                # Breaking out of the candidate loop to restart greedy_add 
                # is safer and simpler.
                break
        
        if improved:
            # Run greedy add again to see if the swap unlocked new additions
            bits, current_size = greedy_add(bits)
            if current_size > best_size:
                best_size = current_size
                best_bits = bits
            # If we improved size, we definitely want to continue.
            # If we just swapped (size same), we continue searching for more swaps.
            # The 'improved' flag handles this.
        else:
            # No swap found. We are stuck in a local optimum of maximal sets.
            # Try a random kick?
            # Remove 2 random elements, try to add 1 or 2?
            # Or just stop?
            # Let's try a simple kick: remove 1 random element, try to add 2 random elements?
            # This is complex. Let's just stop for now, or do a few random restarts?
            # Given the time, let's try a "double swap" or just accept the result.
            # Actually, let's try one more thing: Random Restart with perturbation.
            # If we are stuck, take the current set, remove 5 random elements, 
            # and run greedy_add again.
            
            if current_size > 10:
                # Perturb
                bits_perturbed = bits
                # Remove 5 random elements
                set_elements = [i for i in range(N) if (bits_perturbed >> i) & 1]
                random.shuffle(set_elements)
                for i in range(min(5, len(set_elements))):
                    bits_perturbed &= ~(1 << set_elements[i])
                
                bits_perturbed, size_perturbed = greedy_add(bits_perturbed)
                
                if size_perturbed > current_size:
                    bits = bits_perturbed
                    current_size = size_perturbed
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_bits = bits
                else:
                    # If perturbation didn't help, we might be truly stuck.
                    # Break to save time?
                    # Or try again?
                    # Let's break to avoid infinite loop of useless perturbations.
                    break

    # Final conversion
    result = [i + 1 for i in range(N) if (best_bits >> i) & 1]
    return result

# EVOLVE-BLOCK-END
