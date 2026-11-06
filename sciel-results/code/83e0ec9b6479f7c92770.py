<<<<<<< SEARCH
    def validate_sequence(seq):
        """Verify that a sequence is strictly increasing in [2,128] and all block products are distinct."""
        if not seq:
            return True
        if any(x < 2 or x > 128 for x in seq):
            return False
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i + 1]:
                return False
        # Check all consecutive block products
        products = set()
        for i in range(len(seq)):
            prod_sig = [0] * P
            for j in range(i, len(seq)):
                for k in range(P):
                    prod_sig[k] += sigs_fixed[seq[j]][k]
                prod_tuple = tuple(prod_sig)
                if prod_tuple in products:
                    return False
                products.add(prod_tuple)
        return True
=======
    def validate_sequence(seq):
        """Verify that a sequence is strictly increasing in [2,128] and all block products are distinct."""
        return validate_fast(seq)
>>>>>>> REPLACE

