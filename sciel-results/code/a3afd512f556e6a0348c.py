<<<<<<< SEARCH
    def flexibility_score(m, r, classes, used_moduli):
        """Estimate future constraint cost of choosing residue r for modulus m.
        Lower is better. For each future modulus m2, the blocked residues depend
        on r mod gcd(m, m2). We estimate total blocked count across all future
        moduli, accounting for overlaps with existing classes."""
        score = 0
        for m2 in moduli:
            if m2 in used_moduli or m2 == m:
                continue
            g = gcd(m, m2)
            if g == 1:
                continue
            # (m, r) blocks residues ≡ r (mod g) in m2
            # Check overlap with existing classes
            blocked_by_existing = set()
            for (m3, r3) in classes:
                g3 = gcd(m3, m2)
                if g3 == 1:
                    continue
                # m3 blocks residues ≡ r3 (mod g3) in m2
                # Union with m's blocked set: need to count |{x: x≡r mod g} ∪ {x: x≡r3 mod g3}|
                # This is complex; approximate by just adding g
            score += g
        return score
=======
    def flexibility_score(m, r, classes, used_moduli):
        score = 0
        for m2 in moduli:
            if m2 in used_moduli or m2 == m:
                continue
            g = gcd(m, m2)
            if g == 1:
                continue
            score += m2 // g
        return score
>>>>>>> REPLACE
