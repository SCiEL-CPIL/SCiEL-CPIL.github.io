<<<<<<< SEARCH
    # 3. Structured: Sparse APs
    for _ in range(100):
        A = set()
        while len(A) < n:
            start = random.randint(1, max_val)
            step = random.randint(1, 100)
            for k in range(random.randint(1, 5)):
                val = start + k * step
                if 1 <= val <= max_val:
                    A.add(val)
        A = list(A[:n])
        if is_valid(A):
            candidates.append(A)
=======
    # 3. Structured: Sparse APs (Fixed)
    for _ in range(100):
        A_set = set()
        while len(A_set) < n:
            start = random.randint(1, max_val)
            step = random.randint(1, 100)
            for k in range(random.randint(1, 5)):
                val = start + k * step
                if 1 <= val <= max_val:
                    A_set.add(val)
        A_list = sorted(list(A_set))
        if len(A_list) >= n:
            A_list = A_list[:n]
        if len(A_list) == n and is_valid(A_list):
            candidates.append(A_list)

    # 4. Structured: Union of two APs or AP+GP (Directive strategy)
    for _ in range(200):
        # Choose two blocks
        # Block 1: AP
        start1 = random.randint(1, 500)
        step1 = random.randint(1, 10)
        len1 = random.randint(5, 10)
        block1 = [start1 + i * step1 for i in range(len1)]
        
        # Block 2: AP or GP
        if random.random() < 0.5:
            # Block 2: AP
            start2 = random.randint(500, 2000)
            step2 = random.randint(1, 10)
            len2 = n - len1
            block2 = [start2 + i * step2 for i in range(len2)]
        else:
            # Block 2: GP
            start2 = random.randint(1, 10)
            ratio = random.randint(2, 5)
            len2 = n - len1
            block2 = []
            for i in range(len2):
                val = start2 * (ratio ** i)
                if val > max_val: break
                block2.append(val)
            if len(block2) < len2:
                # Fill remaining with random large numbers to maintain size
                remaining = len2 - len(block2)
                for _ in range(remaining):
                    block2.append(random.randint(max_val//2, max_val))
        
        A_combined = block1 + block2
        if len(A_combined) == n and is_valid(A_combined):
            candidates.append(A_combined)
>>>>>>> REPLACE

