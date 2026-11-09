import itertools
import time

def construct(n):
        if n <= 1:
            return "1"
        
        # Generate standard superpermutation
        # S_n length is sum(i! for i in 1..n)
        
        word = [1]
        for m in range(2, n + 1):
            # Get all permutations of 1..m-1
            perms = list(itertools.permutations(range(1, m)))
            # Order them to maximize overlap?
            # The standard construction uses a specific order.
            # One such order is lexicographic.
            # Let's try lexicographic.
            
            new_word = []
            for p in perms:
                block = list(p) + [m] + list(p)
                # Find max overlap with current new_word
                max_overlap = 0
                min_len = min(len(new_word), len(block))
                for k in range(min_len, -1, -1):
                    if new_word[-k:] == block[:k]:
                        max_overlap = k
                        break
                new_word += block[max_overlap:]
            word = new_word
            
        return "".join(map(str, word))
