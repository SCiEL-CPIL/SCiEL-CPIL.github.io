def construct(n):
       if n == 1:
           return "1"
       if n == 2:
           return "121"
       if n == 3:
           return "123121321"
       
       word = "1"
       for m in range(2, n + 1):
           perms = []
           seen = set()
           # Extract perms
           for i in range(len(word) - m + 2):
               window = word[i:i + m - 1]
               if len(set(window)) == m - 1:
                   if window not in seen:
                       seen.add(window)
                       perms.append(window)
           
           chunks = []
           current_word = "" # We need the current word to find overlaps
           # Actually, we can just build the string if we use a list for the final join
           # But we need the suffix of the current word for overlap.
           # So we keep a string `current_word` but append to a list `chunks`?
           # No, if we append to a list, we can't easily get the suffix for the next overlap
           # unless we join.
           # Given the small size (<6000), string concatenation might be okay,
           # but list + join is safer.
           
           # Let's use a list of strings `parts` and a string `tail` for overlap checking?
           # Or just keep `new_word` as a string. 5000 chars * 5000 appends is 25M char copies.
           # In Python, this might take a second or two. 120s is plenty.
           
           new_word = ""
           for p in perms:
               block = p + str(m) + p
               max_overlap = 0
               limit = min(len(new_word), len(block))
               for o in range(limit, 0, -1):
                   if new_word.endswith(block[:o]): # endswith is faster than slicing?
                       max_overlap = o
                       break
               new_word += block[max_overlap:]
           
           word = new_word
           
       return word
