    def try_remove(seq, idx):
        new_seq = seq[:idx] + seq[idx+1:]
        if is_valid(new_seq):
            return new_seq, True
        return None, False
