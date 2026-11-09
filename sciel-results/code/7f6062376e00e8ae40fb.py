            # Try to add
            success = False
            for _ in range(10): # Try 10 random candidates
                c = random.choice(all_masks)
                if c not in fam_set:
                    if can_add(c, current_fam):
                        current_fam.append(c)
                        fam_set.add(c)
                        success = True
                        break
            
            if not success:
                # Kick
                if len(current_fam) >= 2:
                    idx1 = random.randint(0, len(current_fam) - 1)
                    idx2 = random.randint(0, len(current_fam) - 1)
                    if idx1 == idx2:
                        continue
                    r1 = current_fam.pop(idx1)
                    # Adjust idx2 if needed
                    if idx2 > idx1: idx2 -= 1
                    r2 = current_fam.pop(idx2)
                    fam_set.discard(r1)
                    fam_set.discard(r2)
                    
                    # Greedy refill
                    # Create a shuffled list of all masks not in fam_set
                    candidates = [m for m in all_masks if m not in fam_set]
                    random.shuffle(candidates)
                    for c in candidates:
                        if can_add(c, current_fam):
                            current_fam.append(c)
                            fam_set.add(c)
