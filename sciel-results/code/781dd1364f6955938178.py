cand_set = set(neighbors[essential_words[0], :].tolist())
for ew in essential_words[1:]:
    cand_set.intersection_update(neighbors[ew, :].tolist())
candidates = [c for c in cand_set if not current_in_code[c]]
