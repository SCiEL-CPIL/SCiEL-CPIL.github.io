        uncovered_list = list(range(universe_size))
        uncovered_set = set(uncovered_list)
        
        while uncovered_list:
            # Pick random index
            i = random.randint(0, len(uncovered_list) - 1)
            w = uncovered_list[i]
            
            # Remove w from list and set
            uncovered_list[i] = uncovered_list[-1]
            uncovered_list.pop()
            uncovered_set.remove(w)
            
            candidates = reverse_neighborhoods[w]
            # ... same as before
